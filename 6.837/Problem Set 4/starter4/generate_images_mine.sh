# This is the script we will test your submission with.

SIZE="800 800"
BIN=build/a4

${BIN} -size ${SIZE} -input data/scene01_plane.txt  -output out/mine/a01.png -normals out/mine/a01n.png -depth 8 18 out/mine/a01d.png
${BIN} -size ${SIZE} -input data/scene02_cube.txt   -output out/mine/a02.png -normals out/mine/a02n.png -depth 8 18 out/mine/a02d.png
${BIN} -size ${SIZE} -input data/scene03_sphere.txt -output out/mine/a03.png -normals out/mine/a03n.png -depth 8 18 out/mine/a03d.png
${BIN} -size ${SIZE} -input data/scene04_axes.txt   -output out/mine/a04.png -normals out/mine/a04n.png -depth 8 18 out/mine/a04d.png
${BIN} -size ${SIZE} -input data/scene05_bunny_200.txt -output out/mine/a05.png -normals out/mine/a05n.png -depth 0.8 1.0 out/mine/a05d.png
${BIN} -size ${SIZE} -input data/scene06_bunny_1k.txt -bounces 4 -output out/mine/a06.png -normals out/mine/a06n.png -depth 8 18 out/mine/a06d.png
${BIN} -size ${SIZE} -input data/scene07_arch.txt -bounces 4 -shadows -output out/mine/a07.png -normals out/mine/a07n.png -depth 8 18 out/mine/a07d.png
