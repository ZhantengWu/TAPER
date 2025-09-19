#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片属性分析工具
用于困难子集分层评测
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

def calculate_pca_angle(points):
    """
    使用PCA计算主成分方向角度
    
    Args:
        points: 点集 (N, 2)
        
    Returns:
        float: PCA主成分角度（度）
    """
    if len(points) < 2:
        return 0.0
    
    # 计算质心
    centroid = np.mean(points, axis=0)
    
    # 中心化点集
    centered_points = points - centroid
    
    # 计算协方差矩阵
    cov_matrix = np.cov(centered_points.T)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 获取主成分方向（最大特征值对应的特征向量）
    main_component = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 计算角度
    angle_rad = np.arctan2(main_component[1], main_component[0])
    angle_deg = np.degrees(angle_rad)
    
    # 标准化到0-90度范围
    angle_deg = abs(angle_deg)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    
    return angle_deg

def detect_qr_roi(image):
    """
    使用QRCodeDetector检测QR码ROI，并计算双重角度
    
    Args:
        image: 输入图像
        
    Returns:
        tuple: (success, roi_image, qr_points, roi_info)
    """
    detector = cv2.QRCodeDetector()
    retval, points, _ = detector.detectAndDecode(image)
    
    if points is not None and len(points) > 0:
        # 获取四个角点
        pts = points.reshape(-1, 2).astype(np.int32)
        
        # 方法1: 计算最小外接矩形角度
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 获取矩形的宽高和角度
        (center_x, center_y), (width, height), minarea_angle = rect
        
        # 方法2: 计算PCA主成分角度
        pca_angle = calculate_pca_angle(pts.astype(np.float32))
        
        # 标准化minAreaRect角度到0-45度范围
        minarea_angle = abs(minarea_angle)
        if minarea_angle > 45:
            minarea_angle = 90 - minarea_angle
        
        # 标准化PCA角度到0-45度范围
        if pca_angle > 45:
            pca_angle = 90 - pca_angle
        
        # 双重角度投票机制 - 修复角度计算逻辑
        angle_diff = abs(minarea_angle - pca_angle)
        
        if angle_diff <= 5:  # 两种方法一致（差异≤5度）
            final_angle = (minarea_angle + pca_angle) / 2  # 取平均值
            angle_confidence = "high"
        elif angle_diff <= 15:  # 中等差异
            final_angle = min(minarea_angle, pca_angle)  # 取较小值（更保守）
            angle_confidence = "medium"
        else:  # 差异过大，可能检测不准确
            final_angle = 0.0  # 认为没有明显倾斜
            angle_confidence = "low"
        
        # 创建ROI掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # 提取ROI
        roi = cv2.bitwise_and(image, image, mask=mask)
        
        # 获取边界框用于裁剪
        x, y, w, h = cv2.boundingRect(pts)
        roi_cropped = roi[y:y+h, x:x+w]
        
        roi_info = {
            "center": (center_x, center_y),
            "width": width,
            "height": height,
            "minarea_angle": minarea_angle,
            "pca_angle": pca_angle,
            "final_angle": final_angle,
            "angle_confidence": angle_confidence,
            "angle_diff": angle_diff,
            "min_size": min(width, height),
            "bbox": (x, y, w, h),
            "points": pts
        }
        
        return True, roi_cropped, pts, roi_info
    
    return False, None, None, None

def analyze_image_properties(image_path: str) -> Dict:
    """
    分析单张图片的属性 - 优先基于QR码ROI分析
    
    Args:
        image_path: 图片路径
        
    Returns:
        图片属性字典
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Cannot load image"}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    properties = {
        "image_path": image_path,
        "width": w,
        "height": h
    }
    
    # 尝试使用QRCodeDetector检测QR码ROI
    qr_detected, roi_image, qr_points, roi_info = detect_qr_roi(image)
    
    if qr_detected:
        # 基于QR码ROI进行分析
        properties["qr_detection_method"] = "qr_detector"
        properties["target_width"] = float(roi_info["width"])
        properties["target_height"] = float(roi_info["height"])
        properties["target_min_size"] = float(roi_info["min_size"])
        properties["target_area_ratio"] = float((roi_info["width"] * roi_info["height"]) / (w * h))
        # 移除固定阈值判断，后续用分位数判断
        properties["is_small_target"] = False  # 占位，后续重新计算
        
        # 基于双重角度计算倾斜（minAreaRect + PCA投票）
        properties["minarea_angle"] = float(roi_info["minarea_angle"])
        properties["pca_angle"] = float(roi_info["pca_angle"])
        properties["estimated_angle"] = float(roi_info["final_angle"])
        properties["angle_confidence"] = roi_info["angle_confidence"]
        properties["angle_diff"] = float(roi_info["angle_diff"])
        
        # 倾斜判定：极严格的角度阈值
        if roi_info["angle_confidence"] == "high":
            properties["is_tilted"] = roi_info["final_angle"] > 25  # 高置信度：25度阈值
        elif roi_info["angle_confidence"] == "medium":
            properties["is_tilted"] = roi_info["final_angle"] > 30  # 中等置信度：30度阈值
        else:
            properties["is_tilted"] = False  # 低置信度：不判为倾斜
        
        # 在QR码ROI上计算对比度和模糊度
        if roi_image is not None and roi_image.size > 0:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image
            # 只分析非零区域（实际QR码区域）
            roi_mask = roi_gray > 0
            if np.any(roi_mask):
                roi_pixels = roi_gray[roi_mask]
                std_dev = np.std(roi_pixels)
                mean_val = np.mean(roi_pixels)
                
                properties["contrast_std"] = float(std_dev)
                properties["contrast_mean"] = float(mean_val)
                # 移除固定阈值判断，后续用分位数判断
                properties["is_low_contrast"] = False  # 占位，后续重新计算
                properties["is_dark"] = mean_val < 80  # 保持绝对阈值
                
                # 模糊度检测（在ROI上）
                laplacian_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                properties["blur_score"] = float(laplacian_var)
                properties["is_blurry"] = False  # 占位，后续重新计算
            else:
                # ROI为空，使用默认值
                properties["contrast_std"] = 0.0
                properties["contrast_mean"] = 0.0
                properties["is_low_contrast"] = True
                properties["is_dark"] = True
                properties["blur_score"] = 0.0
                properties["is_blurry"] = True
        else:
            # 退回到全图分析
            std_dev = np.std(gray)
            mean_val = np.mean(gray)
            properties["contrast_std"] = float(std_dev)
            properties["contrast_mean"] = float(mean_val)
            properties["is_low_contrast"] = std_dev < 30 or mean_val < 50
            properties["is_dark"] = mean_val < 80
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            properties["blur_score"] = float(laplacian_var)
            properties["is_blurry"] = laplacian_var < 100
            
    else:
        # QR码检测失败，退回到边缘-轮廓逻辑
        properties["qr_detection_method"] = "contour_fallback"
        
        # 1. 目标尺寸分析（基于轮廓检测）
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的矩形轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓的边界框
            x, y, cw, ch = cv2.boundingRect(largest_contour)
            
            properties["target_width"] = float(cw)
            properties["target_height"] = float(ch)
            properties["target_min_size"] = float(min(cw, ch))
            properties["target_area_ratio"] = float((cw * ch) / (w * h))
            properties["is_small_target"] = False  # 占位，后续重新计算
        else:
            properties["target_width"] = 0.0
            properties["target_height"] = 0.0
            properties["target_min_size"] = 0.0
            properties["target_area_ratio"] = 0.0
            properties["is_small_target"] = False
    
        # 2. 倾斜角度估计（霍夫线变换）
        try:
            # 使用霍夫线变换检测主要线条
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            angles = []
            
            if lines is not None:
                for line in lines[:10]:  # 只取前10条线
                    rho, theta = float(line[0][0]), float(line[0][1])
                    angle = theta * 180 / np.pi
                    # 转换为相对于水平/垂直的角度
                    if angle > 90:
                        angle = angle - 180
                    angles.append(abs(angle))
            
            if angles:
                avg_angle = float(np.mean(angles))
                properties["estimated_angle"] = avg_angle
                properties["is_tilted"] = avg_angle > 35  # 极严格的35度阈值
            else:
                properties["estimated_angle"] = 0.0
                properties["is_tilted"] = False
        except:
            properties["estimated_angle"] = 0.0
            properties["is_tilted"] = False
        
        # 3. 对比度分析（全图）
        std_dev = np.std(gray)
        mean_val = np.mean(gray)
        properties["contrast_std"] = float(std_dev)
        properties["contrast_mean"] = float(mean_val)
        properties["target_area_ratio"] = 0.0  # 轮廓检测无法准确计算面积比
        properties["is_low_contrast"] = False  # 占位，后续重新计算
        properties["is_dark"] = mean_val < 80
        
        # 4. 模糊度检测（全图）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        properties["blur_score"] = float(laplacian_var)
        properties["is_blurry"] = False  # 占位，后续重新计算
    
    # 5. 综合困难度评分 - 降低困难度门槛让模糊图片能进入子集
    difficulty_score = 0
    if properties["is_small_target"]:
        difficulty_score += 1
    if properties["is_tilted"]:
        difficulty_score += 1
    if properties["is_low_contrast"] or properties["is_dark"]:
        difficulty_score += 1
    if properties["is_blurry"]:
        difficulty_score += 1
    
    properties["difficulty_score"] = difficulty_score
    properties["is_difficult"] = difficulty_score >= 1  # 恢复到1个困难因素即可
    
    return properties

def classify_image_difficulty(properties: Dict) -> List[str]:
    """
    根据属性将图片分类到困难子集
    
    Args:
        properties: 图片属性字典
        
    Returns:
        困难类别列表
    """
    categories = []
    
    # 只有真正困难的图片才分到困难子集
    if properties.get("is_difficult", False):
        if properties.get("is_small_target", False):
            categories.append("small_target")
        
        if properties.get("is_tilted", False):
            categories.append("tilted")
        
        if properties.get("is_low_contrast", False) or properties.get("is_dark", False):
            categories.append("low_contrast")
        
        if properties.get("is_blurry", False):
            categories.append("blurry")
    
    # 如果没有困难类别，归为normal
    if not categories:
        categories.append("normal")
    
    return categories

def analyze_dataset_properties(image_dir: str, output_file: str = None) -> Dict:
    """
    分析整个数据集的属性分布
    
    Args:
        image_dir: 图片目录
        output_file: 输出JSON文件路径
        
    Returns:
        数据集分析结果
    """
    image_dir = Path(image_dir)
    image_files = []
    
    # 收集所有图片文件
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
        image_files.extend(image_dir.glob(ext))
    
    if not image_files:
        return {"error": "No images found"}
    
    results = []
    category_counts = {
        "small_target": 0,
        "tilted": 0, 
        "low_contrast": 0,
        "blurry": 0,
        "normal": 0,
        "difficult": 0
    }
    
    print(f"分析 {len(image_files)} 张图片的属性...")
    
    for img_file in image_files:
        try:
            properties = analyze_image_properties(str(img_file))
            if "error" not in properties:
                categories = classify_image_difficulty(properties)
                processed_properties = {
                    "filename": img_file.name,
                    "width": int(properties["width"]),
                    "height": int(properties["height"]),
                    "area": int(properties["width"] * properties["height"]),
                    "is_small_target": bool(properties["is_small_target"]),  # 统一字段名
                    "tilt_angle": float(properties["estimated_angle"]),
                    "is_tilted": bool(properties["is_tilted"]),
                    "contrast_std": float(properties["contrast_std"]),  # 统一字段名
                    "is_low_contrast": bool(properties["is_low_contrast"]),
                    "is_dark": bool(properties.get("is_dark", False)),  # 添加缺失的 is_dark 字段
                    "blur_score": float(properties["blur_score"]),
                    "is_blurry": bool(properties["is_blurry"]),
                    "is_difficult": bool(properties["is_difficult"]),
                    "categories": list(categories),
                    # 添加缺失的字段用于分位数计算
                    "target_area_ratio": float(properties.get("target_area_ratio", 0.0)),
                    "target_min_size": float(properties.get("target_min_size", 0))
                }
                
                # 正确更新分类计数
                for category in categories:
                    if category in category_counts:
                        category_counts[category] += 1
                
                # 修复difficult统计漏洞：检查is_difficult属性
                if properties["is_difficult"]:
                    category_counts["difficult"] += 1
                
                # 如果没有特定困难类别，归为normal
                if not categories:
                    category_counts["normal"] += 1
                
                results.append(processed_properties)
        except Exception as e:
            print(f"分析 {img_file.name} 时出错: {e}")
    
    # 基于分位数重新计算阈值并更新标签
    if results:
        # 收集所有有效数值
        area_ratios = [r["target_area_ratio"] for r in results if r["target_area_ratio"] > 0]
        contrast_stds = [r["contrast_std"] for r in results if r["contrast_std"] > 0]
        blur_scores = [r["blur_score"] for r in results if r["blur_score"] > 0]
        
        # 计算分位数阈值 - 调整模糊度检测
        if area_ratios:
            small_target_threshold = float(np.percentile(area_ratios, 5))  # 面积最小的5%
        else:
            small_target_threshold = 0.002  # 极严格的默认阈值
            
        if contrast_stds:
            low_contrast_threshold = float(np.percentile(contrast_stds, 8))  # 对比度最低的8%
        else:
            low_contrast_threshold = 15  # 极严格的默认阈值
            
        if blur_scores:
            blur_threshold = float(np.percentile(blur_scores, 15))  # 最模糊的15%（放宽）
        else:
            blur_threshold = 100  # 放宽默认阈值
        
        print(f"自适应阈值: 小目标面积比<{small_target_threshold:.4f}, 低对比度<{low_contrast_threshold:.1f}, 模糊度<{blur_threshold:.1f}")
        
        # 重新标记每张图片
        for result in results:
            # 小目标判定：基于面积占比
            if result["target_area_ratio"] > 0:
                result["is_small_target"] = result["target_area_ratio"] < small_target_threshold
            else:
                result["is_small_target"] = result["target_min_size"] < 32  # 退回固定阈值
            
            # 低对比度判定：基于分位数
            if result["contrast_std"] > 0:
                result["is_low_contrast"] = result["contrast_std"] < low_contrast_threshold
            else:
                result["is_low_contrast"] = True  # 无效数据标记为困难
            
            # 模糊判定：基于分位数
            if result["blur_score"] > 0:
                result["is_blurry"] = result["blur_score"] < blur_threshold
            else:
                result["is_blurry"] = True  # 无效数据标记为困难
        
        # 重新计算困难度评分和分类
        for result in results:
            difficulty_score = 0
            categories = []
            
            if result["is_small_target"]:
                difficulty_score += 1
                categories.append("small_target")
            if result["is_tilted"]:
                difficulty_score += 1
                categories.append("tilted")
            if result["is_low_contrast"] or result["is_dark"]:
                difficulty_score += 1
                categories.append("low_contrast")
            if result["is_blurry"]:
                difficulty_score += 1
                categories.append("blurry")
            
            result["difficulty_score"] = difficulty_score
            result["is_difficult"] = difficulty_score >= 1
            result["categories"] = categories if categories else ["normal"]
    
    # 重新统计分类计数
    category_counts = {
        "small_target": 0,
        "tilted": 0, 
        "low_contrast": 0,
        "blurry": 0,
        "normal": 0,
        "difficult": 0
    }
    
    for result in results:
        for category in result["categories"]:
            if category in category_counts:
                category_counts[category] += 1
        
        if result["is_difficult"]:
            category_counts["difficult"] += 1
        
        # 如果没有特定困难类别，归为normal
        if not any(cat in ["small_target", "tilted", "low_contrast", "blurry"] for cat in result["categories"]):
            category_counts["normal"] += 1
    
    dataset_stats = {
        "total_images": len(results),
        "category_distribution": category_counts,
        "category_percentages": {
            cat: (count / len(results) * 100) if results else 0 
            for cat, count in category_counts.items()
        },
        "image_properties": results
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
        print(f"分析结果保存到: {output_file}")
    
    return dataset_stats

def create_difficulty_subsets(properties_file: str, image_dir: str, output_dir: str, exclude_file: str = None) -> Dict:
    """
    根据属性分析创建困难子集
    
    Args:
        properties_file: 属性分析JSON文件
        image_dir: 原始图片目录
        output_dir: 输出目录
        
    Returns:
        子集创建结果
    """
    with open(properties_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建子集目录
    subsets = {
        "small_target": [],
        "tilted": [],
        "low_contrast": [],
        "blurry": [],
        "normal": [],
        "all_difficult": []
    }
    
    for img_props in data["image_properties"]:
        img_name = img_props["filename"]
        categories = img_props.get("categories", [])
        
        for cat in categories:
            if cat in subsets:
                subsets[cat].append(img_name)
        
        if img_props.get("is_difficult", False):
            subsets["all_difficult"].append(img_name)
    
    # 创建子集目录并复制文件
    import shutil
    subset_info = {}
    
    for subset_name, file_list in subsets.items():
        if not file_list:
            continue
            
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(exist_ok=True)
        
        copied_count = 0
        for img_name in file_list:
            src_file = image_dir / img_name
            if src_file.exists():
                dst_file = subset_dir / img_name
                shutil.copy2(src_file, dst_file)
                copied_count += 1
        
        subset_info[subset_name] = {
            "count": copied_count,
            "files": file_list
        }
        
        print(f"创建子集 {subset_name}: {copied_count} 张图片")
    
    return subset_info

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python image_analysis.py <image_dir> [output_json]")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "dataset_properties.json"
    
    results = analyze_dataset_properties(image_dir, output_file)
    
    print("\n数据集属性分布:")
    for cat, count in results["category_distribution"].items():
        percentage = results["category_percentages"][cat]
        print(f"  {cat}: {count} 张 ({percentage:.1f}%)")
