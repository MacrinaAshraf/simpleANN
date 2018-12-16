package simpleANN;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import  java.lang.Math;


public class NeuralNetwork {
	
	 int inputNodesSize , hiddenNodesSize,outputNodesSize;
	 int trainingSetSize;
	 
	 float[][] inputNodes; //input values
	 float[][] hiddenNodes; //hidden layer values
	 float[][] finalOutputNodes; //desired/actual outputs
	
	 float[][]ihWeights; //weights between input layer and hidden layer
	 float[][]hoWeights ; //weights between hidden layer and output layer
	 float[]netIHWeights;
	 float[]netOHWeights;
	 float[][]I;
	 float[][]O;
	 float MSE;
	 double learningRate = 0.8; //can be between 0.1-0.8 (tune it to make better results)
	 
	 //will break when it reaches these acceptable errors otherwise will run all max_epochs.
     double max_acceptable_error = 0.0005; 
     final private double min_acceptable_error = -0.0005;
     final private int max_epochs = 5000; //number of iterations on all the data set.
	 
	 
	
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
			
			inputNodes = new float[trainingSetSize][inputNodesSize];
			hiddenNodes = new float[trainingSetSize][hiddenNodesSize];
			finalOutputNodes = new float[trainingSetSize][outputNodesSize];
			
			for(int i=0;i<trainingSetSize;i++)
			{
				
				tmp = in.readLine().trim().split("\\s+");
				
				for(int j=0, x=0 ;j<tmp.length;j++) {
				
					if(j < inputNodesSize ) {
						//System.out.println(tmp[2]);
						inputNodes[i][j]= Float.parseFloat(tmp[j]);
					}
					
					else {
						finalOutputNodes[i][x]= Float.parseFloat(tmp[j]);
						x++;
					}
				}
				
				
			}
			
			filereader.close();
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
		
		
		for(int i=0;i<hiddenNodesSize;i++) 
		{
			
		   for(int j=0;j<inputNodesSize;j++) 
		   {
			    
			    ihWeights[i][j] = getRandVal();
		   }				
		}
			
		for(int i=0;i<outputNodesSize;i++) 
		{
			
			for(int j=0;j<hiddenNodesSize;j++) 
			{
				hoWeights[i][j] = getRandVal();
			}
		}
		/*
		for(int i=0;i<hiddenNodesSize;i++) 
		{
			 for(int j=0;j<inputNodesSize;j++) 
			{
			  System.out.println(ihWeights[i][j]);
			}
		}	
		
		*/
	}
	
	public double sigmoidFunction(double sigmoidNode){
		return (1/(1+  java.lang.Math.pow(Math.E,-1*sigmoidNode)));
		
	}
	
	public double sigmoidDerivative (double output)
	{
		return output*(1-output);
	}
	
	public void feedForward(int input_data_index){
		 netIHWeights = new float[hiddenNodesSize];
		 netOHWeights = new float[outputNodesSize];
		 I = new float [trainingSetSize][hiddenNodesSize];
		 O = new float [trainingSetSize][outputNodesSize];
		 
		for(int i=0;i<hiddenNodesSize;i++)
		{
			for(int j =0;j<inputNodesSize;j++)
			{
					netIHWeights[i]+=ihWeights[i][j]*inputNodes[input_data_index][j];
			}	
		}
		
		//applying activation function (sigmoid function) 
		for(int i=0; i < hiddenNodesSize ; i++) 
			I[input_data_index][i]=(float) sigmoidFunction( netIHWeights[i] );
				
		
		
        for(int i=0;i<outputNodesSize;i++) 
        {
			for(int j=0;j<hiddenNodesSize;j++)
			{
				 netOHWeights[i] +=hoWeights[i][j]*I[input_data_index][j];
			}
		}
        
        //applying activation function (sigmoid function) 
        for(int i=0; i < outputNodesSize ; i++) 
			O[input_data_index][i]=(float) sigmoidFunction( netOHWeights[i] );
		/* 
		for(int i=0;i<hiddenNodesSize;i++)
			System.out.println("Hidden Marix["+I[i]+"]");
		
		for(int i=0;i<outputNodesSize;i++)
			System.out.println("Output Matrix ["+O[i]+"]");
			
		*/
	}
	
	public void computeHagaShabahElMSE(int input_data_index) {
		
		 for(int i = 0; i < outputNodesSize; i++) {
			 MSE += ( finalOutputNodes[input_data_index][i] - O[input_data_index][i]);
		 }
		 MSE *= (0.5);
		
	}
	
	//Apply Sigmoid function on all inputs and outputs + normalizing values to be 0 < X < 1.
	public void normalization() {
		
		for(int i=0;i<trainingSetSize;i++) {
			
			for(int j=0;j<inputNodesSize;j++) {
				 inputNodes[i][j]/=100; //because there are values exceeding 100.
				 inputNodes[i][j] = (float) sigmoidFunction(inputNodes[i][j]);
			}
				
			
			for(int j=0;j<outputNodesSize;j++) {
				finalOutputNodes[i][j]/=100; //because there are values exceeding 100.
				finalOutputNodes[i][j] = (float) sigmoidFunction(finalOutputNodes[i][j]);
			}
				
			
		}
		
		
	}
	
	public void backPropagation(int input_data_index) {
		
		 float error=0,change_in_delta;
		 float[][] oldWeights = new float[outputNodesSize][hiddenNodesSize];
		 float delta[];
		 
		for(int i=0;i<outputNodesSize;i++) {
				
		    for(int j=0;j<hiddenNodesSize;j++) {
					oldWeights[i][j] = hoWeights[i][j];
				}
		}
		 
		//updating weights for the output-hidden layers.
		for(int i=0 ; i<outputNodesSize;i++) {
			
			delta = new float[outputNodesSize];
			error = finalOutputNodes[input_data_index][i]-O[input_data_index][i];
			delta[i] = (float) ((-1*error)*sigmoidDerivative(O[input_data_index][i]));	
		
		for(int j=0;j<hiddenNodesSize;j++) {
			
			 change_in_delta = (float) (learningRate * delta[i] * I[input_data_index][j]);
			 hoWeights[i][j]= oldWeights[i][j] - change_in_delta;
		   }
		
		}
		
		
		//update weights for the input-hidden layers.
		for(int i=0;i<hiddenNodesSize;i++) {
			
			delta = new float[hiddenNodesSize];
			
			
			for(int y=0;y<outputNodesSize;y++) {
				error += ( ( finalOutputNodes[input_data_index][y]-O[input_data_index][y] ) * oldWeights[y][i] );
			}
			
			for(int j=0;j<inputNodesSize;j++) {
				delta[i] = (float) (sigmoidDerivative(I[input_data_index][i])*error);
				change_in_delta = (float) (learningRate * delta[i]*inputNodes[input_data_index][j]);
				ihWeights[i][j] = ihWeights[i][j] - change_in_delta;
			}
			
		}
		
		
	}
	
	public void weightsToOutputfile() {
		FileWriter writer;
		try {
			writer = new FileWriter("output.txt");
			BufferedWriter bufferedWriter = new BufferedWriter(writer);
			
			bufferedWriter.write(hiddenNodesSize + " " + inputNodesSize + " " + outputNodesSize);
			bufferedWriter.newLine();
			
			for(int i=0;i<hiddenNodesSize;i++) 
			{
				
			   for(int j=0;j<inputNodesSize;j++) 
			   {
				   bufferedWriter.append(Float.toString(ihWeights[i][j]) + " ");
			   }				
			   bufferedWriter.newLine();
			}
			
			bufferedWriter.newLine();
			
			for(int i=0;i<outputNodesSize;i++) 
			{
				for(int j=0;j<hiddenNodesSize;j++) 
				{
					bufferedWriter.append(Float.toString(hoWeights[i][j]) + " ");
				}				
				bufferedWriter.newLine();
			}
			
			bufferedWriter.flush();
			bufferedWriter.close();
			writer.close();
		}
		catch (IOException e) {
			System.out.println("Problem with input/output.");
			e.printStackTrace();
		}
	}
		
	public void train() {
		
		  double percentage_done;
		  initializeWeights();
		  normalization();
		  int p=10;
		  
		 //iterate on all the data set max_epochs times.
		 for(int i=0; i<max_epochs ; i++) {
			//Initialize MSE at the beginning of each iteration on the data set.
			 MSE = 0; 
			 System.out.println("epoche #"+i);
			  //iterate on data set records one by one.
			  for(int j=0;j<trainingSetSize;j++) {
				   
				   //feed forward & back propagate on every record in the data set.
				   feedForward(j); 
				   computeHagaShabahElMSE(j);
				   backPropagation(j);
				  
			        percentage_done = ((float)i/max_epochs)*100;
			        if((int)percentage_done == p) {
			        	System.out.println(String.format("%d %%iterations done", (int)percentage_done)); 
						p+=10;
						System.out.println("data record #" + j);
						for(int x=0;x<outputNodesSize;x++) 
						{
						   System.out.println("Actual output : " + finalOutputNodes[j][x] + " " + "Computed output : "+ O[j][x]);
						}  
						   
			        }    
			  }
			  System.out.println("MSE --> "+ MSE);
			   //if i reached an acceptable error i can break.
			    if(MSE < max_acceptable_error && MSE > min_acceptable_error)
			    	break;
		 }
		
		 weightsToOutputfile();
	}
	
	public void readWeightsFromFile() {
		FileReader filereader;
		BufferedReader in ;
		
		try {
			filereader = new FileReader("output.txt");
			
			in = new BufferedReader(filereader);
			String[] tmp;
			
			tmp = in.readLine().split("\\s+");
			
			hiddenNodesSize = Integer.parseInt(tmp[0]);
			inputNodesSize = Integer.parseInt(tmp[1]);
			outputNodesSize = Integer.parseInt(tmp[2]);
			
			ihWeights = new float[hiddenNodesSize][inputNodesSize] ;
			hoWeights = new float[outputNodesSize][hiddenNodesSize];
			
			for(int i = 0; i < hiddenNodesSize; i++)
			{
				
				tmp = in.readLine().trim().split("\\s+");
				
				for(int j = 0; j < tmp.length; j++) {
						//System.out.println(tmp[2]);
						ihWeights[i][j]= Float.parseFloat(tmp[j]);
					//	System.out.print(ihWeights[i][j] + " ");
				}
				//System.out.println();
			}
			
			tmp[0] = in.readLine();
			
			for(int i = 0; i < outputNodesSize; i++)
			{
				
				tmp = in.readLine().trim().split("\\s+");
				
				for(int j = 0; j < tmp.length; j++) {
						//System.out.println(tmp[2]);
						hoWeights[i][j]= Float.parseFloat(tmp[j]);
					//	System.out.print(hoWeights[i][j] + " ");
				}
				//System.out.println();
			}
			
			filereader.close();
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

	public void Test() {
		readFile("test.txt");
		readWeightsFromFile();
		normalization();
		feedForward(0);
		computeHagaShabahElMSE(0);
		for(int i = 0; i < outputNodesSize; i++)
			System.out.println("Actual output : " + finalOutputNodes[0][i] + " " + "Computed output : "+ O[0][i]);
		System.out.println("MSE --> "+ MSE);
		
	}
}
