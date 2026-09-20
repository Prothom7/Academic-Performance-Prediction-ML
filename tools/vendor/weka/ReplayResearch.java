import java.io.*;
import java.nio.charset.StandardCharsets;
import weka.core.*;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.converters.ConverterUtils.DataSource;

/** Replays a preserved classifier and independently refits its frozen settings.
 * Nominal labels are checked by unique English prefix to tolerate historical
 * bilingual display corruption; no values, rows or nominal ordering are changed.
 */
public class ReplayResearch {
  private static String key(String value) {
    int paren = value.indexOf('(');
    return (paren >= 0 ? value.substring(0,paren) : value).trim();
  }
  private static Instances read(String path) throws Exception {
    BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(path), StandardCharsets.UTF_8));
    Instances data = new Instances(reader);
    reader.close();
    data.setClassIndex(data.numAttributes()-1);
    return data;
  }
  private static Instances align(Instances source, Instances header) {
    if (source.numAttributes()!=header.numAttributes()) throw new IllegalArgumentException("Attribute count mismatch");
    Instances aligned = new Instances(header,source.numInstances());
    for (Instance row: source) {
      double[] values = new double[source.numAttributes()];
      for (int j=0;j<values.length;j++) {
        Attribute a=source.attribute(j), b=header.attribute(j);
        if (!a.name().equals(b.name()) || a.type()!=b.type()) throw new IllegalArgumentException("Attribute mismatch: "+a.name());
        if (row.isMissing(j)) values[j]=Utils.missingValue();
        else if (a.isNumeric()) values[j]=row.value(j);
        else {
          String wanted=key(row.stringValue(j));
          int index=-1;
          for (int k=0;k<b.numValues();k++) if (key(b.value(k)).equals(wanted)) {
            if (index!=-1) throw new IllegalArgumentException("Ambiguous label: "+wanted);
            index=k;
          }
          if (index==-1) throw new IllegalArgumentException("Unknown label: "+wanted);
          values[j]=index;
        }
      }
      aligned.add(new DenseInstance(row.weight(),values));
    }
    return aligned;
  }
  public static void main(String[] args) throws Exception {
    Object[] stored=SerializationHelper.readAll(args[0]);
    Classifier frozen=(Classifier)stored[0];
    Instances header=(Instances)stored[1];
    Instances sourceTrain=read(args[1]);
    Instances train=align(sourceTrain,header), test=align(read(args[2]),header);
    Instances converted=DataSource.read(args[1]);
    converted.setClassIndex(converted.numAttributes()-1);
    converted=align(converted,header);
    int differenceCount=0;
    for(int i=0;i<train.numInstances();i++) for(int k=0;k<train.numAttributes();k++) {
      double original=train.instance(i).value(k), convertedValue=converted.instance(i).value(k);
      if(original!=convertedValue && !(Double.isNaN(original) && Double.isNaN(convertedValue))) differenceCount++;
    }
    RandomForest refit=new RandomForest();
    refit.setOptions(Utils.splitOptions("-P 100 -attribute-importance -I 100 -num-slots 1 -K 12 -M 1.0 -V 0.001 -S 1"));
    refit.buildClassifier(train);
    PrintWriter output=new PrintWriter(new OutputStreamWriter(new FileOutputStream(args[3]),StandardCharsets.UTF_8));
    output.println("row_id,actual,frozen_predicted,refit_predicted,frozen_probability,refit_probability");
    for (int i=0;i<test.numInstances();i++) {
      Instance row=test.instance(i);
      output.println(i+","+(int)row.classValue()+","+(int)frozen.classifyInstance(row)+","+(int)refit.classifyInstance(row)+","+frozen.distributionForInstance(row)[1]+","+refit.distributionForInstance(row)[1]);
    }
    output.close();
    System.out.println("Header labels matched by unique English prefix; nominal indices aligned to preserved header.");
    System.out.println("Train N="+train.numInstances()+"; Test N="+test.numInstances());
    System.out.println("Input-cell differences between explicit UTF-8 reader and CLI DataSource converter: "+differenceCount);
    System.out.println("Data-dependent RNG with original symbolic header: "+train.getRandomNumberGenerator(1).nextLong());
    System.out.println("Data-dependent RNG with UTF-8 source symbolic header: "+sourceTrain.getRandomNumberGenerator(1).nextLong());
    System.out.println("Frozen parameters: "+Utils.joinOptions(refit.getOptions()));
  }
}
