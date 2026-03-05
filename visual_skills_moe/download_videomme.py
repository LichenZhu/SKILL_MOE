import os
import zipfile
from glob import glob
from huggingface_hub import snapshot_download
from tqdm import tqdm

# 配置
REPO_ID = "lmms-lab/Video-MME"
LOCAL_DIR = "./benchmarks/data/video_mme"
ALLOW_PATTERNS = [
    "videos_chunked_*.zip",  # 视频分卷
    "subtitle.zip",          # 字幕
    "*.parquet",             # 元数据
    "*.json",                # 元数据
    "README.md"
]

def unzip_file(zip_path, extract_to):
    """解压 zip 文件并显示进度"""
    print(f"📦 正在解压: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            try:
                zf.extract(member, extract_to)
            except zipfile.error as e:
                print(f"❌ 解压错误 {member.filename}: {e}")

def main():
    print("🚀 开始从 Hugging Face 下载 Video-MME...")
    print(f"📂 目标目录: {os.path.abspath(LOCAL_DIR)}")
    
    # 1. 下载文件
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=ALLOW_PATTERNS,
            tqdm_class=tqdm
        )
        print("✅ 下载完成！")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return

    # 2. 解压视频
    video_dir = os.path.join(LOCAL_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    zip_files = sorted(glob(os.path.join(LOCAL_DIR, "videos_chunked_*.zip")))
    zip_files += sorted(glob(os.path.join(LOCAL_DIR, "subtitle.zip")))
    
    print(f"🔍 发现 {len(zip_files)} 个压缩包，准备解压...")
    
    for zip_f in zip_files:
        unzip_file(zip_f, video_dir)
        # 解压后如果想节省空间，可以取消下面这行的注释来删除 zip 包
        # os.remove(zip_f)

    print("\n🎉 Video-MME 数据集准备就绪！")
    print(f"视频存放位置: {video_dir}")

if __name__ == "__main__":
    # 需要安装依赖: pip install huggingface_hub tqdm
    main()
    