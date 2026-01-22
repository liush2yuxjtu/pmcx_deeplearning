#!/bin/bash

# 获取所有可用的 GPU 节点
NODES=$(sinfo -h -o %n)

if [ -z "$NODES" ]; then
    echo "没有可用的节点"
    exit 1
fi

echo "当前集群节点状态:"
sinfo -o "%20n %15C %m %25G %10T" -e
for node in $NODES; do
    echo -n "$node: "
    scontrol show node $node | grep -E 'FreeMem' | awk '{print $3}'
done

echo "----------------------------------------"
# 遍历所有节点
for NODE in $NODES; do
    #echo -e "\n节点: $NODE"
    
    # 获取节点的可用 CPU 和内存信息
    CPUS_ALLOC=$(scontrol show node=$NODE | grep CPUAlloc | awk '{print $1}' | cut -d'=' -f2)
    CPUS_TOTAL=$(scontrol show node=$NODE | grep CPUTot | awk '{print $3}' | cut -d'=' -f2)
    AVAIL_CPUS=$((CPUS_TOTAL - CPUS_ALLOC))

    MEM_ALLOC=$(scontrol show node=$NODE | grep AllocMem | awk '{print $2}' | cut -d'=' -f2)
    MEM_TOTAL=$(scontrol show node=$NODE | grep RealMemory | awk '{print $1}' | cut -d'=' -f2)
    AVAIL_MEM=$((MEM_TOTAL - MEM_ALLOC))
    #AVAIL_MEM=$(scontrol show node=$NODE | grep FreeMem | awk '{print $3}' | cut -d'=' -f2)

    # 获取可用的 GPU 数量
    AVAIL_GPU=$(scontrol show node=$NODE | grep Gres | grep -o 'gpu:[0-9]' | cut -d':' -f2)

    #echo "可用 CPU: $AVAIL_CPUS / $CPUS_TOTAL (已用: $CPUS_ALLOC)"
    #echo "可用内存: $AVAIL_MEM / $MEM_TOTAL MB (已用: $MEM_ALLOC MB)"
    #echo "可用 GPU: $AVAIL_GPU"

    USED_CPU=$((AVAIL_CPUS/2))
    USED_MEM=$((AVAIL_MEM/2))
    
    #echo "推荐的 srun 命令:"
    if timeout 2 srun --immediate -w $NODE --gres=gpu:1 echo "hello" &>/dev/null; then
    #if timeout 10 srun --immediate --pty -w $node --gres=gpu:1 -c $USED_CPU --mem=${USED_MEM}M echo "hello" &>/dev/null; then
        # if used_cpu>18 , print another line with --cpus-per-gpu=18 instead of $USED_CPU
        # if mem-per-gpu>12000, print another line with --mem-per-gpu=12000 instead of $USED_MEM
        echo "srun --pty -w $NODE --gres=gpu:1 --cpus-per-gpu=$USED_CPU --mem-per-gpu=${USED_MEM}M /bin/bash"
        if [ $USED_CPU -gt 18 ]; then
            echo "srun --pty -w $NODE --gres=gpu:1 --cpus-per-gpu=18 --mem-per-gpu=${USED_MEM}M /bin/bash"
        fi
        if [ $USED_MEM -gt 12000 ]; then
            echo "srun --pty -w $NODE --gres=gpu:1 --cpus-per-gpu=$USED_CPU --mem-per-gpu=12000M /bin/bash"
        fi
        if [ $USED_MEM -gt 12000 -a $USED_CPU -gt 18 ]; then
            echo "srun --pty -w $NODE --gres=gpu:1 --cpus-per-gpu=18 --mem-per-gpu=12000M /bin/bash"
        fi
        echo "----------------------------------------"
    fi
    # echo "srun --pty -w $NODE --gres=gpu:1 -c $USED_CPU --mem=${USED_MEM}M /bin/bash"
    # echo "----------------------------------------"
done

squeue -u $USER