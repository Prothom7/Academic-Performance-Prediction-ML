import weka.core.SerializationHelper;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
public class InspectModel {
  public static void main(String[] args) throws Exception {
    Object[] objects = SerializationHelper.readAll(args[0]);
    System.out.println("Serialized objects: " + objects.length);
    for (Object object: objects) {
      System.out.println(object.getClass().getName());
      if (object instanceof Instances) {
        Instances header = (Instances)object;
        Instances current = DataSource.read(args[1]);
        current.setClassIndex(current.numAttributes()-1);
        System.out.println("Header difference: " + header.equalHeadersMsg(current));
        System.out.println(header.toString());
      }
    }
  }
}
