package me.zhongjun.weka;
import java.io.File;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;


public class KNNCV {  
	 public static int folds = 5;
	 public static void run(String fileName) {
	       // TODO Auto-generated method stub
	       Instances trainIns = null;
	       Instances testIns = null;
	       IBk cfs = null;
	      
	      
	       try{
	          
	           /* 1
	            * read data
	            */
	           File file= new File(fileName);
	           ArffLoader loader = new ArffLoader();
	           loader.setFile(file);
	           trainIns = loader.getDataSet();
	           trainIns.setClassIndex(trainIns.numAttributes()-1); //在使用样本之前一定要首先设置instances的classIndex，否则在使用instances对象是会抛出异常
       
	           /* 2
	            * creat classifier
	            */
	           cfs = (IBk)Class.forName("weka.classifiers.lazy.IBk").newInstance();
	           cfs.setKNN(1);//设置邻居的个数
	           cfs.buildClassifier(trainIns);
	           
	           
	           //evaluate classifier with cross validation
	            Evaluation eval=new Evaluation(trainIns);

	            //first supply the classifier
	            //then the training data
	            //number of folds
	            //random seed
	            eval.crossValidateModel(cfs, trainIns, folds, new Random(1));
	            
	            System.out.println("Percent correct: "+
	                               Double.toString(eval.pctCorrect()));
	          
	       }catch(Exception e){
	           e.printStackTrace();
	       }
	    }
	 
}