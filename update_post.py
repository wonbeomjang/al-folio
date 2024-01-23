import os
from glob import glob

if __name__ == "__main__":
    user_path = os.path.expanduser("~")
    obsidian_post_path = user_path + "/Library/Mobile Documents/iCloud~md~obsidian/Documents/Wonbeom-Jang/1. paper/"
    obsidian_image_path = user_path + "/Library/Mobile Documents/iCloud~md~obsidian/Documents/Wonbeom-Jang/0. assets/post/image/"
    obsidian_image_path = glob(obsidian_post_path + "/**/*", recursive=True)
    blog_path = os.getcwd()

    for md_file_name in os.listdir(obsidian_post_path):
        if md_file_name.startswith("."):
            continue

        md_f = open(os.path.join(obsidian_post_path, md_file_name), "r", encoding="utf-8")
        blog_f = open(os.path.join(blog_path, "_posts", md_file_name), "w", encoding="utf-8")

        for line in md_f.readlines():
            line = line.replace("![[", '\n<p align="center"><img src="/assets/post/image/')
            line = line.replace("]]", '" width="80%"></p>\n')
            line = line.replace(" $", "$$")
            line = line.replace("$ ", "$$")
            blog_f.write(line)

        md_f.close()
        blog_f.close()