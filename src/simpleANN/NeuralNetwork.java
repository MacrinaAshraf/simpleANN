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
	 float[]netIHWeights;
	 float[]netOHWeights;
	 float []I;
	 float []O;
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
		ihWeights = new float[hiddenNodesSize][inputNodesSize] ;
		hoWeights = new float[outputNodesSize][hiddenNodesSize];
		
		
		for(int i=0;i<hiddenNodesSize;i++) {
			
		 for(int j=0;j<inputNodesSize;j++) {
			    
			    ihWeights[i][j] = getRandVal();
		 }				
		}
			
		for(int i=0;i<outputNodesSize;i++) {
			
			for(int j=0;j<hiddenNodesSize;j++) {
				hoWeights[i][j] = getRandVal();
			}
		}
		for(int i=0;i<hiddenNodesSize;i++) {
			
			 for(int j=0;j<inputNodesSize;j++) {
			System.out.println(ihWeights[i][j]);
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
	
	public void matrixMultiplication(){
		 netIHWeights = new float[hiddenNodesSize];
		 netOHWeights = new float[outputNodesSize];
		 I = new float [hiddenNodesSize];
		 O = new float[outputNodesSize];
		 
		for(int i=0;i<hiddenNodesSize;i++)
		{
			for(int j =0;j<inputNodesSize;j++)
			{
					netIHWeights[i]+=ihWeights[i][j]*inputNodes.get(j);
			}	
		}
		
		for(int i=0; i < hiddenNodesSize ; i++) //applying activation function (sigmoid function) 
			I[i]=(float) sigmoidFunction( netIHWeights[i] );
				
		
		
		
		
        for(int i=0;i<outputNodesSize;i++) 
        {
			for(int j=0;j<hiddenNodesSize;j++)
			{
				 netOHWeights[i] +=hoWeights[i][j]*I[j];
			}
		}
        
        for(int i=0; i < outputNodesSize ; i++) //applying activation function (sigmoid function) 
			O[i]=(float) sigmoidFunction( netOHWeights[i] );
		 
		for(int i=0;i<hiddenNodesSize;i++)
			System.out.println("Hiden Marix["+I[i]+"]");
		
		for(int i=0;i<outputNodesSize;i++)
			System.out.println("Outpu Matrix ["+O[i]+"]");
	}
	
}
