package me.zhongjun.weka;
  
  
/** 
 * desc:试试Weka的决策树类 
 * <code>J48Test</code> 
 * @version 1.0 2011/12/13 
 * @author chenwq 
 * 
 */  
import java.io.File;  
import java.io.IOException;  
  





import weka.classifiers.Classifier;  
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;  
import weka.core.Instance;
import weka.core.Instances;  
import weka.core.converters.ArffLoader;  
  
public class SMOTest {  
  
    public static void run(String trainFile, String testFile) {  
	    try{
	        File inputFile = new File(trainFile);//训练语料文件  
	        ArffLoader atf = new ArffLoader();   
	        atf.setFile(inputFile);
	        Instances trainIns = atf.getDataSet(); // 读入训练文件 
	        trainIns.setClassIndex(trainIns.numAttributes()-1);
	        
	        inputFile = new File(testFile);//测试语料文件  
	        atf.setFile(inputFile);            
	        Instances testIns = atf.getDataSet(); // 读入测试文件  
	        testIns.setClassIndex(testIns.numAttributes()-1);
	
	        // classifier
	        weka.classifiers.functions.SMO scheme = new weka.classifiers.functions.SMO();
	        scheme.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\"")); 
	        scheme.buildClassifier(trainIns);
	        
	        Instance testInst;
	        Evaluation testingEvaluation = new Evaluation(testIns);
	        int length = testIns.numInstances();
	        for (int i =0; i < length; i++) {
	           testInst = testIns.instance(i);
	           //通过这个方法来用每个测试样本测试分类器的效果
	           testingEvaluation.evaluateModelOnceAndRecordPrediction(
	        		   scheme, testInst);
	        }
	        System.out.println( "分类器的正确率：" + (1- testingEvaluation.errorRate()));
	    }catch(Exception e){
	        e.printStackTrace();
	    }
    }  
  
}  