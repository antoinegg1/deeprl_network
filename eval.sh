
export CUDA_VISIBLE_DEVICES=0

RESULT_ROOT="/home/lichangye/deeprl_network/result"

# 遍历 RESULT_ROOT 目录下的所有一级子目录
for dir in "$RESULT_ROOT"/*/; do
  # ${dir%/} 用来去掉最后的斜杠，保证路径整洁
  dir="${dir%/}"
#要求dir中包含ic3net
  [[ ! "$dir" =~ ic3net ]] && continue

  # 如果目录不存在或不是目录（极端情况下），跳过
  [[ ! -d "$dir" ]] && continue

  echo "==> Evaluating directory: $dir"
  python3 main.py --base-dir "$dir" evaluate --evaluation-seeds 237
  echo    # 空行分隔，让日志更清晰
done
