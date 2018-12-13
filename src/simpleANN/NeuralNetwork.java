package simpleANN;

import  java.lang.Math;

public class NeuralNetwork {
	
	
	
	public void readFile(String filePath) {
	}

	public double sigmoidFunction(double sigmoidNode){
		return (1/(1+  java.lang.Math.pow(Math.E,-1*sigmoidNode)));
		
	}
	
	public double sigmoidDerivative (double output)
	{
		return output*(1-output);
	}
	
	
}
