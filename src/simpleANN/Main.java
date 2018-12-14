package simpleANN;

public class Main {

	public static void main(String[] args) {
		
		NeuralNetwork n = new NeuralNetwork();
		
		n.readFile("SimpleExample.txt");
		
		n.initializeWeights();
		n.matrixMultiplication();
	}

}
