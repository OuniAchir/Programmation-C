#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Fonction d'activation et sa dérivée
double sigmoid(double x) { return 1 / (1 + exp(-x)); } //x= la somme pondérée des entrées
double dSigmoid(double x) { return x * (1 - x); }

//Fonction d'initialisation des poids
double init_weight() { return ((double)rand())/((double)RAND_MAX); }//Normalisée la valeur générée pour qu'elle soit entre [0,1]

// Shuffle the dataset : Cette pratique peut améliorer la convergence du modèle et l'aider à généraliser plus efficacement.
void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
//Nombre d'entrées
#define numInputs 2
//Nombre de neuods de la couche cachée
#define numHiddenNodes 2
//Nombre de sortie
#define numOutputs 1
//Nombre de l'ensemble d'entrainement
#define numTrainingSets 4


int main (void) {
    //Pas d'apprentissage
    const double lr = 0.1f;

    //Tableau d'une couche cachée a 2 neurones
    double hiddenLayer[numHiddenNodes];
    //Tableau de couche de sortie a une seule neurone
    double outputLayer[numOutputs];


    //Biais de la couche cachée
    double hiddenLayerBias[numHiddenNodes];
    //Biais de la couche de sortie
    double outputLayerBias[numOutputs];


    //Poid des neurones de la couche cachée qui sera de longeur de nombre d'entrée et le deuxieme parametre l'index du neurones de la couche cachée
    double hiddenWeights[numInputs][numHiddenNodes];
    //Poid des neurones de couche de sortie longeur nombre de neurones de la couche cachée et lindex de ma sortie
    double outputWeights[numHiddenNodes][numOutputs];

    //Matrice des données d'entrainement i=nombre d'ensemble d'entrainement et j=nombre d'entrée
    double training_inputs[numTrainingSets][numInputs] = {{0.0f,0.0f},
                                                          {1.0f,0.0f},
                                                          {0.0f,1.0f},
                                                          {1.0f,1.0f}};
    //Matrice de sorties i=nombre d'nesemble d'entrainement et j=nombre de sortie
    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                            {0.0f},
                                                            {0.0f},
                                                            {1.0f}};

    //initialisation des poids de la couche cachée
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }

    //Initialiation des biais de la couche cachée
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j=0; j<numOutputs; j++) {
            outputWeights[i][j] = init_weight();
        }
    }

    //initualisation des bais de la couche de sortie
    for (int i=0; i<numOutputs; i++) {
        outputLayerBias[i] = init_weight();
    }


    int trainingSetOrder[] = {0,1,2,3};

    //Nombre d'épochs pour le quel on entraine notre réseau de neurones
    int numberOfEpochs = 10000;

    //Boucle d'entrainement de notre réseau de neurones
    for(int epochs=0; epochs < numberOfEpochs; epochs++) {

        //Modifier l'ordre de notre ensemble d'entrainement
        shuffle(trainingSetOrder,numTrainingSets);

        // Cycle through each of the training set elements
        for (int x=0; x<numTrainingSets; x++) {

            int i = trainingSetOrder[x];

            // Forward propagation

            //Calcule de la fonction d'activation pour chaque neurone de la couche cachée
            for (int j=0; j<numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                 for (int k=0; k<numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            //Calcule de la fonction d'activation pour la couche de sortie
            for (int j=0; j<numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            //Affichage des résultats de la Forward propagation
            printf ("Entree:%g %g        Sortie predite:%g        Sortie reelle: %g\n",
                    training_inputs[i][0], training_inputs[i][1],
                    outputLayer[0], training_outputs[i][0]);



            // Backpropagation

            //Calcule de l'erreur entre la sortie predite et la sortie reelle pour la couche de sortie
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
            }

            //Calcule de l'erreur entre la sortie predite et la sortie reelle pour la couche cachée
            double deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    errorHidden += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = errorHidden * dSigmoid(hiddenLayer[j]);
            }

            //Mise a jour des poids de sortie
            for (int j=0; j<numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k=0; k<numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            //Mise a jour des poids de la couche cachée
            for (int j=0; j<numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for(int k=0; k<numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    //Affichage des poids finals
    fputs ("Final Hidden Weights\n[ ", stdout);
    for (int j=0; j<numHiddenNodes; j++) {
        fputs ("[ ", stdout);
        for(int k=0; k<numInputs; k++) {
            printf ("%f ", hiddenWeights[k][j]);
        }
        fputs ("] ", stdout);
    }

    fputs ("]\nFinal Hidden Biases\n[ ", stdout);
    for (int j=0; j<numHiddenNodes; j++) {
        printf ("%f ", hiddenLayerBias[j]);
    }

    fputs ("]\nFinal Output Weights", stdout);
    for (int j=0; j<numOutputs; j++) {
        fputs ("[ ", stdout);
        for (int k=0; k<numHiddenNodes; k++) {
            printf ("%f ", outputWeights[k][j]);
        }
        fputs ("]\n", stdout);
    }

    fputs ("Final Output Biases\n[ ", stdout);
    for (int j=0; j<numOutputs; j++) {
        printf ("%f ", outputLayerBias[j]);

    }

    fputs ("]\n", stdout);

    return 0;
}
