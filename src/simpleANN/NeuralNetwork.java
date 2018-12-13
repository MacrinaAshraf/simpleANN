package simpleANN;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import  java.lang.Math;


public class NeuralNetwork {
	
	 int inputNodesSize , hiddenNodesSize,outputNodesSize;
	 int trainingSetSize;
	 
	 List<Float> inputNodes = new ArrayList<Float>();
	 List<Float> hiddenNodes = new ArrayList<Float>();
	 List<Float> finlaOutputNodes = new ArrayList<Float>();
	
	 float[][]ihWeights;
	 public float[][] hoWeights ;
	  List<Float> OutputNodes = new ArrayList<Float>();
	 
	
	 
	
	public void readFile(String filePath) {

		
		FileReader filereader;
		BufferedReader in ;
		
		try {
			filereader = new FileReader(filePath);
			
			in = new BufferedReader(filereader);
			String[] tmp;
			
			tmp=in.readLine().split("\\s+");
			
			inputNodesSize = Integer.parseInt(tmp[0]);
			hiddenNodesSize = Integer.parseInt(tmp[1]);
			outputNodesSize = Integer.parseInt(tmp[2]);
			
			tmp[0] = in.readLine();
			
			trainingSetSize = Integer.parseInt(tmp[0]);
			
			for(int i=0;i<trainingSetSize;i++)
			{
				
				tmp = in.readLine().split("\\s+");
				
				for(int j=0, x=0 ;j<tmp.length;j++) {
					
					if(j < inputNodesSize ) {
						inputNodes.add(j, Float.parseFloat(tmp[j]));
					}
					
					else {
						finlaOutputNodes.add(x, Float.parseFloat(tmp[j]));
						x++;
					}
				}
				
				
			}
			
			
		}
	    
		catch (FileNotFoundException e) {
			System.out.println("File is not found.");
			e.printStackTrace();
		}
		catch (IOException e) {
			System.out.println("Problem with input/output.");
			e.printStackTrace();
		}	
		
		
		
	}
	
	private float getRandVal() {
		
		 Random r = new Random();
		 float x = r.nextFloat();
		 return x;
		 
	}
	
	public void initializeWeights() {
		ihWeights = new float[inputNodesSize][hiddenNodesSize] ;
		hoWeights = new float[hiddenNodesSize][outputNodesSize];
		
		
		for(int i=0;i<inputNodesSize;i++) {
			
		 for(int j=0;j<hiddenNodesSize;j++) {
			    
			    ihWeights[i][j] = getRandVal();
			  
		 }		
				
		}
		
		
		for(int i=0;i<hiddenNodesSize;i++) {
			
			for(int j=0;j<outputNodesSize;j++) {
				hoWeights[0][0] = getRandVal();
			}
		}
		
		
	}
	
	
	
	

	public double sigmoidFunction(double sigmoidNode){
		return (1/(1+  java.lang.Math.pow(Math.E,-1*sigmoidNode)));
		
	}
	
	public double sigmoidDerivative (double output)
	{
		return output*(1-output);
	}
	
	
}
