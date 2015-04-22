package me.zhongjun.weka;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import wlsvm.WLSVM;

//Did not finish yet
//Does not work!!!!!!!!!
public class SVMCV {  
	 public static void run(String[] args) {

	       try{
	          arff2libsvm("cjlt123-fea10.arff");

	       }catch(Exception e){
	           e.printStackTrace();
	       }
	 }
	 public static String arff2libsvm(String fileName){
		 BufferedReader br;
		 
		try {
			br = new BufferedReader(new FileReader(new File(fileName)));
			
			BufferedWriter out=new BufferedWriter(new FileWriter("temp"));
			
			String line = br.readLine();
			CharSequence cs = "data";
			while(line!=null){
			 if(line.contains(cs)){
				 line = br.readLine();
				 while(line!=null){
					 
					 out.write(transformFeatures(line));
					 out.write('\n');
					 out.flush();
					 line = br.readLine();
				 }
			 }else{
				 line = br.readLine();
			 }
			}
			br.close();
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		 System.out.println("done!");
		 return "temp";
	 }
	 
	 private static String transformFeatures(String features){
		 char ch[] = features.toCharArray();
		 int i = ch.length-1;
		 while(ch[i] != ' '){
			 i--;
		 }
		 String last = new String(ch,i+1,ch.length-i-2);
		 CharSequence cs = ".";
		 if(last.contains(cs)){
			 String res = "1\t"+new String(ch,1,ch.length-2).replace(' ', '\t').replace(',', '\t');
			 return transformDataFormat(res);
		 }else{
			 String cls = new String(ch,i+1,ch.length-(i+2));
			 while(ch[i] != ','){
				 i--;
			 }
			 String res = new String(ch,1,i-1);
			 res = cls+'\t'+res.replace(' ', '\t').replace(',', '\t');
			 return transformDataFormat(res);
		 }
	 }
	 
	 private static String transformDataFormat(String data){
		 String strs[] = data.split("\t");
		 for(int i=1; i<strs.length; i++){
			 if(i%2==1){
				 strs[i] = "\t"+(Integer.valueOf(strs[i])+1);
			 }else{
				 strs[i] = "\t"+((int)(Double.valueOf(strs[i])*10000.0));
			 }
		 }
		 StringBuilder sb = new StringBuilder();
		 for(String s:strs){
			 sb.append(s);
		 }
		 return sb.toString();
	 }
	 
}