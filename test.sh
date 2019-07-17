echo "amazon..."
./exp partition/code/amazon-2.vertices partition/code/amazon-2.edges 735292 3523247 539775 1761624 4 >> result/amazon.txt
echo "DBLP..."
./exp partition/code/dblp.vertices partition/code/dblp.edges 933258 3353618 585631 1676810 4 >> result/dblp.txt
echo "livejournal..."
./exp partition/code/livejournal.vertices partition/code/livejournal.edges 4846609 42851237 3287720 21425619 4 >> result/livejournal.txt
echo "roadNet..."
./exp partition/code/RoadNet.vertices  partition/code/RoadNet.edges    1965206  2766607 1233852 1383304 4 >> result/roadnet.txt
echo "wiki..."
./exp partition/code/wiki.vertices  partition/code/wiki.edges      2394385  4659565 1213436 2329783 4 >> result/wiki.txt
echo "youtube..."
./exp partition/code/youtube.vertices partition/code/youtube.edges 1134890  2987624 630533 1493813 4 >> result/youtube.txt
echo "stanford..."
./exp partition/code/stanford.vertices  partition/code/stanford.edges  281903 1992636   184271 996319 4 >> result/stanford.txt