package simpleANN;

public class Main {

	public static void main(String[] args) {
		
		NeuralNetwork n = new NeuralNetwork();
		
		n.readFile("train.txt");	
		//n.initializeWeights();
		//n.train();
		n.Test();
	}

}
