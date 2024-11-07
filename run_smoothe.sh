python launch.py --acyclic --dataset realistic --method smoothe --repeat 3 --greedy_ini
python launch.py --acyclic --dataset synthetic --method smoothe --repeat 3 --greedy_ini
python launch.py --acyclic --dataset rover --method smoothe --repeat 1
python launch.py --acyclic --dataset tensat --method smoothe --repeat 1

python launch.py --acyclic --dataset realistic --method smoothe --repeat 1 --cost quad
python launch.py --acyclic --dataset realistic --method smoothe --repeat 1 --cost mlp