#!/bin/bash
# 安装项目自定义git hooks
SRC_HOOK_DIR="./git-hooks"
DST_HOOK_DIR=".git/hooks"

if [ ! -d "${SRC_HOOK_DIR}" ];then
    echo "error: git‑hooks目录不存在"
    exit 1
fi

cp ${SRC_HOOK_DIR}/* ${DST_HOOK_DIR}/
chmod +x ${DST_HOOK_DIR}/*
echo "✅ git pre‑commit大文件检测钩子安装完成"
