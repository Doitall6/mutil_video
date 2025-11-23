#!/usr/bin/env python3
"""
图片尺寸调整工具
将图片调整为 ReID 模型所需的 256(高) x 128(宽) 尺寸
保持宽高比，不失真，使用黑边填充
"""
import cv2
import numpy as np
import sys
import os
from pathlib import Path

def resize_image(input_path, output_path=None, width=128, height=256):
    """
    调整图片尺寸（保持宽高比，不失真）
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径（可选）
        width: 目标宽度，默认128
        height: 目标高度，默认256
    """
    # 读取图片
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"❌ 错误: 无法读取图片 {input_path}")
        return False
    
    h, w = img.shape[:2]
    print(f"📷 原始尺寸: {w}x{h} (宽x高)")
    
    # 计算缩放比例（保持宽高比）
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 调整大小 - 使用高质量的插值方法
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 创建目标尺寸的黑色画布
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 计算居中位置
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    
    # 将调整后的图片放到画布中央
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    print(f"✓ 调整后尺寸: {new_w}x{new_h} (缩放比例: {scale:.3f})")
    print(f"✓ 最终尺寸: {width}x{height} (居中填充)")
    
    # 确定输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_256x128{ext}"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ 创建目录: {output_dir}")
    
    # 保存图片
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✓ 已保存到: {output_path}")
    
    return True

def resize_directory(input_dir, output_dir=None, width=128, height=256):
    """
    批量调整目录中所有图片的尺寸
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选）
        width: 目标宽度
        height: 目标高度
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ 错误: 目录不存在 {input_dir}")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # 查找所有图片
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print("="*60)
    
    # 确定输出目录
    if output_dir is None:
        output_dir = str(input_path / 'resized')
    
    # 处理每张图片
    success_count = 0
    for img_file in image_files:
        print(f"\n处理: {img_file.name}")
        output_path = os.path.join(output_dir, img_file.name)
        
        if resize_image(str(img_file), output_path, width, height):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"✓ 完成! 成功处理 {success_count}/{len(image_files)} 张图片")
    print(f"输出目录: {output_dir}")

def main():
    if len(sys.argv) < 2:
        print("="*60)
        print("图片尺寸调整工具 - ReID 专用 (256高 x 128宽)")
        print("="*60)
        print("\n使用方法:")
        print("  1. 调整单张图片:")
        print(f"     python {sys.argv[0]} <input_image> [output_image]")
        print()
        print("  2. 批量调整目录中的所有图片:")
        print(f"     python {sys.argv[0]} --dir <input_directory> [output_directory]")
        print()
        print("示例:")
        print(f"  # 单张图片")
        print(f"  python {sys.argv[0]} person.jpg")
        print(f"  python {sys.argv[0]} person.jpg query/person_256x128.jpg")
        print()
        print(f"  # 批量处理")
        print(f"  python {sys.argv[0]} --dir ./original_images")
        print(f"  python {sys.argv[0]} --dir ./original_images ./query")
        print("="*60)
        sys.exit(1)
    
    # 批量处理模式
    if sys.argv[1] == '--dir':
        if len(sys.argv) < 3:
            print("❌ 错误: 请指定输入目录")
            print(f"使用方法: python {sys.argv[0]} --dir <input_directory> [output_directory]")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        resize_directory(input_dir, output_dir)
    
    # 单文件处理模式
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(input_path):
            print(f"❌ 错误: 文件不存在 {input_path}")
            sys.exit(1)
        
        resize_image(input_path, output_path)

if __name__ == "__main__":
    main()
