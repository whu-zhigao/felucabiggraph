#rm -rf ./src/*.o  ./exp  
#rm -rf obj
#make

#small example with output
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/output.vertices ./example/output.edges 6 8 3 2 4 >file.txt  2>&1
#without output
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/output.vertices ./example/dataset-4/output.edges 6 8 3 2 4  ./example/sample_graph.txt

#wiki
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/wiki-talk.vertices ./example/wiki-talk.edges 2394385 5021410  618287  1164892 4 >> ./txt/file.txt 

#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/wiki-talk.vertices ./example/dataset-4/wiki-talk.edges 2394385 5021410  618287  1164892 4  
#amazon

#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/amazon.vertices ./example/dataset-4/amazon.edges 735322 5158012 356275 880813 4 

#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/amazon.vertices ./example/amazon.edges 735322 5158012 356275 880813 4 

echo "amazon..."
cuda-memcheck ./exp ./partition/code/amazon-2.vertices ./partition/code/amazon-2.edges 735292 3523247 539854 1761624 4
echo "dblp..."
cuda-memcheck ./exp ./partition/code/dblp.vertices ./partition/code/dblp.edges 933258 3353618 585794 1676809 4
echo "livejournal..."
cuda-memcheck ./exp ./partition/code/livejournal.vertices ./partition/code/livejournal.edges 4846609 42851237 3288764 21425619 4
echo "RoadNet..."
cuda-memcheck ./exp ./partition/code/RoadNet.vertices ./partition/code/RoadNet.edges 1965206 2766607 1233147 1383304 4
echo "Stanford..."
cuda-memcheck ./exp ./partition/code/stanford.vertices ./partition/code/stanford.edges 281903 1992636 184513 996318 4
echo "wiki-talk..."
cuda-memcheck ./exp ./partition/code/wiki.vertices ./partition/code/wiki.edges 2394385 4659565 1213436 2329783 4
echo "youtube..."
cuda-memcheck ./exp ./partition/code/youtube.vertices ./partition/code/youtube.edges 1134890 2987624 630533 1493813 4