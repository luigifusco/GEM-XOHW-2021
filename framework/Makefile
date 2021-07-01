make: losses.lib transforms.lib

losses.lib:
	gcc -O3 -fPIC -shared -o losses.lib losses.cpp

transforms.lib:
	gcc -O3 -fPIC -shared -o transforms.lib transforms.cpp

.PHONY: clean

clean:
	rm *.lib
