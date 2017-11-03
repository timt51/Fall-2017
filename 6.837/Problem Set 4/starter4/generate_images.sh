# This is the script we will test your submission with.

SIZE="800 800"
BIN=sample_solution/athena/a4

${BIN} -size ${SIZE} -input data/scene01_plane.txt  -output out/sample/a01.png -normals out/sample/a01n.png -depth 8 18 out/sample/a01d.png
${BIN} -size ${SIZE} -input data/scene02_cube.txt   -output out/sample/a02.png -normals out/sample/a02n.png -depth 8 18 out/sample/a02d.png
${BIN} -size ${SIZE} -input data/scene03_sphere.txt -output out/sample/a03.png -normals out/sample/a03n.png -depth 8 18 out/sample/a03d.png
${BIN} -size ${SIZE} -input data/scene04_axes.txt   -output out/sample/a04.png -normals out/sample/a04n.png -depth 8 18 out/sample/a04d.png
${BIN} -size ${SIZE} -input data/scene05_bunny_200.txt -output out/sample/a05.png -normals out/sample/a05n.png -depth 0.8 1.0 out/sample/a05d.png
${BIN} -size ${SIZE} -input data/scene06_bunny_1k.txt -bounces 4 -output out/sample/a06.png -normals out/sample/a06n.png -depth 8 18 out/sample/a06d.png
${BIN} -size ${SIZE} -input data/scene07_arch.txt -bounces 4 -shadows -output out/sample/a07.png -normals out/sample/a07n.png -depth 8 18 out/sample/a07d.png
