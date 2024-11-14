all: bilayer

precompiled:
	mkdir -p precompiled

bilayer: precompiled
	manim -qh multilayer_network.py BiLayer
	mv media/images/multilayer_network/BiLayer_Manim*.png precompiled/

clean:
	rm -rf media