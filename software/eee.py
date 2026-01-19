import os
import argparse
import yaml


def generate_image_yaml(input_dir, output_yaml, prefix="./rknn/", extensions=None):
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 确保扩展名小写
    extensions = {ext.lower() for ext in extensions}

    image_paths = []

    for filename in sorted(os.listdir(input_dir)):
        _, ext = os.path.splitext(filename)
        if ext.lower() in extensions:
            full_path = os.path.join(prefix, filename)
            image_paths.append(full_path)

    # 写入 YAML
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(image_paths, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ 成功生成 {len(image_paths)} 条记录到: {output_yaml}")
    print(f"📁 示例条目:\n  - {image_paths[0] if image_paths else '（无）'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 ./test/images/<文件名> 格式的 YAML 图片列表")
    parser.add_argument("input_dir", help="输入图片文件夹路径")
    parser.add_argument("output_yaml", help="输出的 YAML 文件路径")
    parser.add_argument("--prefix", default="./test/images/", help="路径前缀 (默认: ./test/images/)")
    parser.add_argument("--ext", nargs="*", default=[".jpg", ".png"],
                        help="要包含的文件扩展名 (默认: .jpg .png)")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"❌ 错误: 输入路径不是有效文件夹: {args.input_dir}")
        exit(1)

    generate_image_yaml(
        input_dir=args.input_dir,
        output_yaml=args.output_yaml,
        prefix=args.prefix.rstrip('/') + '/',  # 确保以 / 结尾
        extensions=args.ext
    )