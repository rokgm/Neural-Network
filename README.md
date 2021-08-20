# Neural-Network
Projekt za predmet računalništvo. Nevronska mreža z vzvratnim razširjanjem napake (en skriti sloj).
Uporabil bom STOCHASTIC GRADIENT DESCENT.

# Uporaba
Glej `example.ipynb` in docstringe v `neural_n.NeuralNetwork`.

## Podatki za učenje in evaluacijo
Network za učenje sprejme seznam tuplov (parov (input, output)).
Input in output morata biti vektorja. Output v obliki [0,0,...,1,...,0] (Če label predstavlja index se uporabi: neural_n.NeuralNetwork.one_hot_encoder).
V primeru `example.ipynb` je input še normaliziran.

Podatke je lažje naložiti z `numpy.loadtxt()`. V primeru `example.ipynb` se s to metodo pojavi `MemoryError: cannot allocate array memory`. Podatki so zato naloženi z zanko, kjer se lahko tudi izbere koliko primerov hočemo naložiti.

V `example.ipynb` je uporabljena `mnist database` v formatu `.csv`. V njej je 60000 primerov slik napisanih številk (28 x 28 pixels) in vsaka ima label, kater številko slika predstavlja. Potem je navoljo 10000 testnih primerov v enakem formatu. Database je dostopen na https://www.python-course.eu/neural_network_mnist.php.

## Učenje in evaluacija
Najprej se naredi struktura networka z bin.neural_n.NeuralNetwork v `example.ipynb` imported as `NN`.
Z `NeuralNetwork.train` se network nauči na primerih za učenje. Network se lahko tudi shrani z `NeuralNetwork.save_network`.

Evaluacija z `NeuralNetwork.evaluate` na testnih primerih. Točnost napovedi `NeuralNetwork.accuracy`.

## Loading Network iz file-a in napovedi na primeru
Network shranjen z `NeuralNetwork.save_network` se naloži z `NeuralNetwork.load_network`.

# Viri:
- https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- https://builtin.com/data-science/gradient-descent
- https://en.wikipedia.org/wiki/Backpropagation
- https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
- https://mlfromscratch.com/neural-network-tutorial/#/
- https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd
- https://www.heatonresearch.com/2017/06/01/hidden-layers.html
- https://www.python-course.eu/neural_network_mnist.php