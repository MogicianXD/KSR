nohup python -u run_KSR.py --data m15cat --type A --epochs 300 --lr 0.002 --cuda 5 --pretrain > m15A0 2>&1 &
nohup python -u run_KSR.py --data m15cat --type B --epochs 300 --lr 0.002 --cuda 5 --pretrain > m15B0 2>&1 &

#nohup python -u run_KSR.py --data m15cat --type A --epochs 300 --lr 0.002 --cuda 4 > m15A3 2>&1 &
#nohup python -u run_KSR.py --data m15cat --type B --epochs 300 --lr 0.002 --cuda 5 > m15B3 2>&1 &

#nohup python -u run_KSR.py --data year15 --type A --epochs 300 --lr 0.002 --cuda 2 --pretrain > y15A2 2>&1 &
#nohup python -u run_KSR.py --data year15 --type A --epochs 300 --lr 0.002 --cuda 4 > y15A 2>&1 &
#nohup python -u run_KSR.py --data year15 --type B --epochs 300 --lr 0.002 --cuda 5 > y15B 2>&1 &
#nohup python -u run_KSR.py --data year15 --type B --epochs 300 --lr 0.002 --cuda 3 --pretrain > y15B2 2>&1 &
