import java.util.Random;

import javax.swing.JFrame;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

/**
 * Write a description of class ClassficationWeka here.
 *
 * @author (your name)
 * @version (a version number or a date)
 */
public class ClassficationWeka
{
    
    
    public static void main(String args[]) {
        try {
            DataSource source  = new DataSource("data/zoo.arff");
            Instances data = source.getDataSet();
            //System.out.println(data.numInstances()+ " instances loaded");
            //System.out.println(data.toString());

            Remove remove = new Remove();
            String[] opts = new String[] {"-R", "1"};
            remove.setOptions(opts);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            //System.out.println(data.toString());

            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            Ranker search = new Ranker();
            AttributeSelection attselect = new AttributeSelection();
            attselect.setEvaluator(eval);
            attselect.setSearch(search);
            attselect.SelectAttributes(data);

            int[] indices = attselect.selectedAttributes();
            //System.out.println(Utils.arrayToString(indices));

            J48 tree = new J48();
            String[] options = new String[1];
            options[0] = "-U";
            tree.setOptions(options);
            tree.buildClassifier(data);
            //System.out.println(tree);
            //double result = tree.classifyInstance(myUnicorn);
            //System.out.println(data.classAttribute().value((int) result));

            // Vector to house attributes of new animals.
            java.util.List<double[]> animalsToClassify = new java.util.Vector<double[]>();
            
            // Unicorn: expecting mammal
            double[] unicorn = new double[data.numAttributes()];
            unicorn[0] = 1.0; //hair {false, true}
            unicorn[1] = 0.0; //feathers {false, true}
            unicorn[2] = 0.0; //eggs {false, true}
            unicorn[3] = 1.0; //milk {false, true}
            unicorn[4] = 0.0; //airborne {false, true}
            unicorn[5] = 0.0; //aquatic {false, true}
            unicorn[6] = 0.0; //predator {false, true}
            unicorn[7] = 0.0; //toothed {false, true}
            unicorn[8] = 1.0; //backbone {false, true}
            unicorn[9] = 1.0; //breathes {false, true}
            unicorn[10] = 1.0; //venomous {false, true}
            unicorn[11] = 0.0; //fins {false, true}
            unicorn[12] = 4.0; //legs INTEGER [0,9]
            unicorn[13] = 1.0; //tail {false, true}
            unicorn[14] = 1.0; //domestic {false, true}
            unicorn[15] = 0.0; //catsize {false, true}
            animalsToClassify.add(unicorn);

            // Shark: expecting fish
            double[] shark = new double[data.numAttributes()];
            shark[0] = 0.0; //hair {false, true}
            shark[1] = 0.0; //feathers {false, true}
            shark[2] = 0.0; //eggs {false, true}
            shark[3] = 0.0; //milk {false, true}
            shark[4] = 0.0; //airborne {false, true}
            shark[5] = 1.0; //aquatic {false, true}
            shark[6] = 1.0; //predator {false, true}
            shark[7] = 1.0; //toothed {false, true}
            shark[8] = 1.0; //backbone {false, true}
            shark[9] = 1.0; //breathes {false, true}
            shark[10] = 0.0; //venomous {false, true}
            shark[11] = 1.0; //fins {false, true}
            shark[12] = 0.0; //legs INTEGER [0,9]
            shark[13] = 1.0; //tail {false, true}
            shark[14] = 0.0; //domestic {false, true}
            shark[15] = 0.0; //catsize {false, true}
            animalsToClassify.add(shark);

            //Tarantula: expecting invertibre
            double[] tarantula = new double[data.numAttributes()];
            tarantula[0] = 1.0; //hair {false, true}
            tarantula[1] = 0.0; //feathers {false, true}
            tarantula[2] = 1.0; //eggs {false, true}
            tarantula[3] = 0.0; //milk {false, true}
            tarantula[4] = 0.0; //airborne {false, true}
            tarantula[5] = 0.0; //aquatic {false, true}
            tarantula[6] = 1.0; //predator {false, true}
            tarantula[7] = 1.0; //toothed {false, true}
            tarantula[8] = 0.0; //backbone {false, true}
            tarantula[9] = 1.0; //breathes {false, true}
            tarantula[10] = 1.0; //venomous {false, true}
            tarantula[11] = 0.0; //fins {false, true}
            tarantula[12] = 8.0; //legs INTEGER [0,9]
            tarantula[13] = 0.0; //tail {false, true}
            tarantula[14] = 0.0; //domestic {false, true}
            tarantula[15] = 0.0; //catsize {false, true}
            animalsToClassify.add(tarantula);

            // Chameleon: expecting reptile
            double[] chameleon = new double[data.numAttributes()];
            chameleon[0] = 0.0; //hair {false, true}
            chameleon[1] = 0.0; //feathers {false, true}
            chameleon[2] = 1.0; //eggs {false, true}
            chameleon[3] = 0.0; //milk {false, true}
            chameleon[4] = 0.0; //airborne {false, true}
            chameleon[5] = 0.0; //aquatic {false, true}
            chameleon[6] = 1.0; //predator {false, true}
            chameleon[7] = 1.0; //toothed {false, true}
            chameleon[8] = 1.0; //backbone {false, true}
            chameleon[9] = 1.0; //breathes {false, true}
            chameleon[10] = 0.0; //venomous {false, true}
            chameleon[11] = 0.0; //fins {false, true}
            chameleon[12] = 4.0; //legs INTEGER [0,9]
            chameleon[13] = 1.0; //tail {false, true}
            chameleon[14] = 1.0; //domestic {false, true}
            chameleon[15] = 0.0; //catsize {false, true}
            animalsToClassify.add(chameleon);

            // Human: exxpecting mammal
            double[] human = new double[data.numAttributes()];
            human[0] = 1.0; //hair {false, true}
            human[1] = 0.0; //feathers {false, true}
            human[2] = 0.0; //eggs {false, true}
            human[3] = 1.0; //milk {false, true}
            human[4] = 0.0; //airborne {false, true}
            human[5] = 0.0; //aquatic {false, true}
            human[6] = 1.0; //predator {false, true}
            human[7] = 1.0; //toothed {false, true}
            human[8] = 1.0; //backbone {false, true}
            human[9] = 1.0; //breathes {false, true}
            human[10] = 0.0; //venomous {false, true}
            human[11] = 0.0; //fins {false, true}
            human[12] = 2.0; //legs INTEGER [0,9]
            human[13] = 0.0; //tail {false, true}
            human[14] = 1.0; //domestic {false, true}
            human[15] = 0.0; //catsize {false, true}
            animalsToClassify.add(human);

            // Ant: expecting insect
            double[] ant = new double[data.numAttributes()];
            ant[0] = 0.0; //hair {false, true}
            ant[1] = 0.0; //feathers {false, true}
            ant[2] = 1.0; //eggs {false, true}
            ant[3] = 0.0; //milk {false, true}
            ant[4] = 0.0; //airborne {false, true}
            ant[5] = 0.0; //aquatic {false, true}
            ant[6] = 0.0; //predator {false, true}
            ant[7] = 0.0; //toothed {false, true}
            ant[8] = 0.0; //backbone {false, true}
            ant[9] = 1.0; //breathes {false, true}
            ant[10] = 0.0; //venomous {false, true}
            ant[11] = 0.0; //fins {false, true}
            ant[12] = 6.0; //legs INTEGER [0,9]
            ant[13] = 0.0; //tail {false, true}
            ant[14] = 0.0; //domestic {false, true}
            ant[15] = 0.0; //catsize {false, true}
            animalsToClassify.add(ant);

            // Grasshopper: expecting insect
            double[] grasshopper = new double[data.numAttributes()];
            grasshopper[0] = 0.0; //hair {false, true}
            grasshopper[1] = 0.0; //feathers {false, true}
            grasshopper[2] = 1.0; //eggs {false, true}
            grasshopper[3] = 0.0; //milk {false, true}
            grasshopper[4] = 1.0; //airborne {false, true}
            grasshopper[5] = 0.0; //aquatic {false, true}
            grasshopper[6] = 0.0; //predator {false, true}
            grasshopper[7] = 0.0; //toothed {false, true}
            grasshopper[8] = 0.0; //backbone {false, true}
            grasshopper[9] = 0.0; //breathes {false, true}
            grasshopper[10] = 0.0; //venomous {false, true}
            grasshopper[11] = 0.0; //fins {false, true}
            grasshopper[12] = 6.0; //legs INTEGER [0,9]
            grasshopper[13] = 0.0; //tail {false, true}
            grasshopper[14] = 0.0; //domestic {false, true}
            grasshopper[15] = 0.0; //catsize {false, true}
            animalsToClassify.add(grasshopper);

            // Salamander: expecting amphibian
            double[] salamander = new double[data.numAttributes()];
            salamander[0] = 0.0; //hair {false, true}
            salamander[1] = 0.0; //feathers {false, true}
            salamander[2] = 1.0; //eggs {false, true}
            salamander[3] = 0.0; //milk {false, true}
            salamander[4] = 0.0; //airborne {false, true}
            salamander[5] = 1.0; //aquatic {false, true}
            salamander[6] = 1.0; //predator {false, true}
            salamander[7] = 1.0; //toothed {false, true}
            salamander[8] = 1.0; //backbone {false, true}
            salamander[9] = 1.0; //breathes {false, true}
            salamander[10] = 0.0; //venomous {false, true}
            salamander[11] = 0.0; //fins {false, true}
            salamander[12] = 4.0; //legs INTEGER [0,9]
            salamander[13] = 1.0; //tail {false, true}
            salamander[14] = 0.0; //domestic {false, true}
            salamander[15] = 0.0; //catsize {false, true}
            animalsToClassify.add(salamander);
            
            // Start time
            long start = System.currentTimeMillis();
            
            for (int i = 0; i < animalsToClassify.size(); i++) {

                Instance myAnimal = new DenseInstance(1.0, animalsToClassify.get(i));
                myAnimal.setDataset(data);
                double result = tree.classifyInstance(myAnimal);
                System.out.println("ANIMAL TYPE = " + data.classAttribute().value((int) result));

            }
            
            Classifier cl = new J48();
            Evaluation eval_roc = new Evaluation(data);
            eval_roc.crossValidateModel(cl, data, 10, new Random(1), new Object[] {});

            // End time
            long end = System.currentTimeMillis();
            System.out.println("Execution time: " + (end-start) + "ms");
            
            /*
            //System.out.println(eval_roc.toSummaryString());
            double[][] confusionMatrix = eval_roc.confusionMatrix();
            //System.out.println(eval_roc.toMatrixString());
            */
            
        }
        catch(Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }

}
