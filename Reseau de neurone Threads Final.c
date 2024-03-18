#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

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


struct ThreadData {
    int startIdx;
    int endIdx;
    double (*hiddenLayer)[numHiddenNodes];
    double (*outputLayer)[numOutputs];
    double (*hiddenLayerBias)[numHiddenNodes];
    double (*outputLayerBias)[numOutputs];
    double (*hiddenWeights)[numInputs][numHiddenNodes];
    double (*outputWeights)[numHiddenNodes][numOutputs];
    double (*training_inputs)[numTrainingSets][numInputs];
    double (*training_outputs)[numTrainingSets][numOutputs];
};

void* trainRange(void* arg) {

    //Pas d'apprentissage
    const double lr = 0.1f;

    struct ThreadData* data = (struct ThreadData*)arg;

     int trainingSetOrder[] = {0,1,2,3};

    //Nombre d'épochs pour le quel on entraine notre réseau de neurones
    int numberOfEpochs = 100000;

    for(int epochs=0; epochs < numberOfEpochs; epochs++) {

        //Modifier l'ordre de notre ensemble d'entrainement
        shuffle(trainingSetOrder,numTrainingSets);

        // Cycle through each of the training set elements
        for (int x=0; x<numTrainingSets; x++) {

            int i = trainingSetOrder[x];

            // Forward propagation

            //Calcule de la fonction d'activation pour chaque neurone de la couche cachée
            for (int j=0; j<numHiddenNodes; j++) {
                double activation = (*data->hiddenLayerBias)[j];
                 for (int k=0; k<numInputs; k++) {
                    activation += (*data->training_inputs)[i][k] * (*data->hiddenWeights)[k][j];
                }
                (*data->hiddenLayer)[j] = sigmoid(activation);
            }

            //Calcule de la fonction d'activation pour la couche de sortie
            for (int j=0; j<numOutputs; j++) {
                double activation = (*data->outputLayerBias)[j];
                for (int k=0; k<numHiddenNodes; k++) {
                    activation += (*data->hiddenLayer)[k] * (*data->outputWeights)[k][j];
                }
                (*data->outputLayer)[j] = sigmoid(activation);
            }


            //Affichage des résultats de la Forward propagation
            if(epochs==99999){
            printf ("Entree:%g %g        Sortie predite:%g        Sortie reelle: %g\n",
                    (*data->training_inputs)[i][0], (*data->training_inputs)[i][1],
                    (*data->outputLayer)[0], (*data->training_outputs)[i][0]);
            }
            // Backpropagation

            //Calcule de l'erreur entre la sortie predite et la sortie reelle pour la couche de sortie
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = ((*data->training_outputs)[i][j] - (*data->outputLayer)[j]);
                deltaOutput[j] = errorOutput * dSigmoid((*data->outputLayer)[j]);
            }

            //Calcule de l'erreur entre la sortie predite et la sortie reelle pour la couche cachée
            double deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    errorHidden += deltaOutput[k] * (*data->outputWeights)[j][k];
                }
                deltaHidden[j] = errorHidden * dSigmoid((*data->hiddenLayer)[j]);
            }

            //Mise a jour des poids de sortie
            for (int j=0; j<numOutputs; j++) {
                (*data->outputLayerBias)[j] += deltaOutput[j] * lr;
                for (int k=0; k<numHiddenNodes; k++) {
                    (*data->outputWeights)[k][j] += (*data->hiddenLayer)[k] * deltaOutput[j] * lr;
                }
            }

            //Mise a jour des poids de la couche cachée
            for (int j=0; j<numHiddenNodes; j++) {
                (*data->hiddenLayerBias)[j] += deltaHidden[j] * lr;
                for(int k=0; k<numInputs; k++) {
                    (*data->hiddenWeights)[k][j] += (*data->training_inputs)[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

     //Affichage des poids finals
    fputs ("Final Hidden Weights\n[ ", stdout);
    for (int j=0; j<numHiddenNodes; j++) {
        fputs ("[ ", stdout);
        for(int k=0; k<numInputs; k++) {
            printf ("%f ", (*data->hiddenWeights)[k][j]);
        }
        fputs ("] ", stdout);
    }

    fputs ("]\nFinal Hidden Biases\n[ ", stdout);
    for (int j=0; j<numHiddenNodes; j++) {
        printf ("%f ", (*data->hiddenLayerBias)[j]);
    }

    fputs ("]\nFinal Output Weights", stdout);
    for (int j=0; j<numOutputs; j++) {
        fputs ("[ ", stdout);
        for (int k=0; k<numHiddenNodes; k++) {
            printf ("%f ", (*data->outputWeights)[k][j]);
        }
        fputs ("]\n", stdout);
    }

    fputs ("Final Output Biases\n[ ", stdout);
    for (int j=0; j<numOutputs; j++) {
        printf ("%f ", (*data->outputLayerBias)[j]);

    }

    fputs ("]\n", stdout);
    return NULL;
}

int main (void) {

    //Tableau d'une couche cachée a 2 neurones
    double hiddenLayer[numHiddenNodes];
    //Tableau de couche de sortie a une seule neurone
    double outputLayer[numOutputs];


    //Biais de la couche cachée AND
    double hiddenLayerBias_AND[numHiddenNodes];
    //Biais de la couche de sortie AND
    double outputLayerBias_AND[numOutputs];

    //Biais de la couche cachée OR
    double hiddenLayerBias_OR[numHiddenNodes];
    //Biais de la couche de sortie OR
    double outputLayerBias_OR[numOutputs];

    //Biais de la couche cachée XOR
    double hiddenLayerBias_XOR[numHiddenNodes];
    //Biais de la couche de sortie XOR
    double outputLayerBias_XOR[numOutputs];

    //Poid des neurones de la couche cachée qui sera de longeur de nombre d'entrée et le deuxieme parametre l'index du neurones de la couche cachée
    double hiddenWeights_AND[numInputs][numHiddenNodes];
    //Poid des neurones de couche de sortie longeur nombre de neurones de la couche cachée et lindex de ma sortie
    double outputWeights_AND[numHiddenNodes][numOutputs];

    //Poid des neurones de la couche cachée qui sera de longeur de nombre d'entrée et le deuxieme parametre l'index du neurones de la couche cachée
    double hiddenWeights_OR[numInputs][numHiddenNodes];
    //Poid des neurones de couche de sortie longeur nombre de neurones de la couche cachée et lindex de ma sortie
    double outputWeights_OR[numHiddenNodes][numOutputs];

    //Poid des neurones de la couche cachée qui sera de longeur de nombre d'entrée et le deuxieme parametre l'index du neurones de la couche cachée
    double hiddenWeights_XOR[numInputs][numHiddenNodes];
    //Poid des neurones de couche de sortie longeur nombre de neurones de la couche cachée et lindex de ma sortie
    double outputWeights_XOR[numHiddenNodes][numOutputs];

    //Matrice des données d'entrainement i=nombre d'ensemble d'entrainement et j=nombre d'entrée
    double training_inputs[numTrainingSets][numInputs] = {{0.0f,0.0f},
                                                          {1.0f,0.0f},
                                                          {0.0f,1.0f},
                                                          {1.0f,1.0f}};

    //Matrice de sorties i=nombre d'nesemble d'entrainement et j=nombre de sortie AND
    double training_outputs_AND[numTrainingSets][numOutputs] = {{0.0f},
                                                            {0.0f},
                                                            {0.0f},
                                                            {1.0f}};

    //Matrice de sorties i=nombre d'nesemble d'entrainement et j=nombre de sortie OR
    double training_outputs_OR[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {1.0f}};

    //Matrice de sorties i=nombre d'nesemble d'entrainement et j=nombre de sortie XOR
    double training_outputs_XOR[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}};
    //initialisation des poids de la couche  AND
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights_AND[i][j] = init_weight();
        }
    }

    //initialisation des poids de la couche  OR
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights_OR[i][j] = init_weight();
        }
    }

    //initialisation des poids de la couche  XOR
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights_XOR[i][j] = init_weight();
        }
    }

    //Initialiation des biais de la couche cachée AND
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias_AND[i] = init_weight();
        for (int j=0; j<numOutputs; j++) {
            outputWeights_AND[i][j] = init_weight();
        }
    }

    //Initialiation des biais de la couche cachée OR
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias_OR[i] = init_weight();
        for (int j=0; j<numOutputs; j++) {
            outputWeights_OR[i][j] = init_weight();
        }
    }

    //Initialiation des biais de la couche cachée XOR
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias_XOR[i] = init_weight();
        for (int j=0; j<numOutputs; j++) {
            outputWeights_XOR[i][j] = init_weight();
        }
    }

    //initualisation des bais de la couche de sortie AND
    for (int i=0; i<numOutputs; i++) {
        outputLayerBias_AND[i] = init_weight();
    }

    //initualisation des bais de la couche de sortie OR
    for (int i=0; i<numOutputs; i++) {
        outputLayerBias_OR[i] = init_weight();
    }

    //initualisation des bais de la couche de sortie XOR
    for (int i=0; i<numOutputs; i++) {
        outputLayerBias_XOR[i] = init_weight();
    }

    int trainingSetOrder[] = {0,1,2,3};

    // Nombre de threads
    int numThreads = 3;

    // Tableau des threads
    pthread_t threads[numThreads];

    // tableau pour les données de chaque threads
    struct ThreadData threadData[numThreads];

    // Divisé l'ensemble d'entrainement sur les  threads
    int chunkSize = numTrainingSets / numThreads;
    int remainder = numTrainingSets % numThreads;
    int startIdx = 0;

    for (int i = 0; i < numThreads; i++) {
        threadData[i].startIdx = startIdx;
        threadData[i].endIdx = startIdx + chunkSize + (i < remainder ? 1 : 0);


        if (i == 0) { // Premier thread pour AND
            threadData[i].hiddenLayer = &hiddenLayer;
            threadData[i].outputLayer = &outputLayer;
            threadData[i].hiddenLayerBias = &hiddenLayerBias_AND;
            threadData[i].outputLayerBias = &outputLayerBias_AND;
            threadData[i].hiddenWeights = &hiddenWeights_AND;
            threadData[i].outputWeights = &outputWeights_AND;
            threadData[i].training_inputs = &training_inputs;
            threadData[i].training_outputs = &training_outputs_AND;
        } else if (i==1){ // Deuxième thread pour OR
            threadData[i].hiddenLayer = &hiddenLayer;
            threadData[i].outputLayer = &outputLayer;
            threadData[i].hiddenLayerBias = &hiddenLayerBias_OR;
            threadData[i].outputLayerBias = &outputLayerBias_OR;
            threadData[i].hiddenWeights = &hiddenWeights_OR;
            threadData[i].outputWeights = &outputWeights_OR;
            threadData[i].training_inputs = &training_inputs;
            threadData[i].training_outputs = &training_outputs_OR;
        }else { // Troisiéme thread pour XOR
            threadData[i].hiddenLayer = &hiddenLayer;
            threadData[i].outputLayer = &outputLayer;
            threadData[i].hiddenLayerBias = &hiddenLayerBias_XOR;
            threadData[i].outputLayerBias = &outputLayerBias_XOR;
            threadData[i].hiddenWeights = &hiddenWeights_XOR;
            threadData[i].outputWeights = &outputWeights_XOR;
            threadData[i].training_inputs = &training_inputs;
            threadData[i].training_outputs = &training_outputs_XOR;
        }
        startIdx = threadData[i].endIdx;
    }

    // Create and run threads
    for (int i = 0; i < numThreads; i++) {
        pthread_create(&threads[i], NULL, trainRange, &threadData[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < numThreads; i++) {
           // if (i == 0) {printf("************************************************************************************************************************\n");}
            //else printf("\n************************************************************************************************************************\n");
        pthread_join(threads[i], NULL);
    }

    return 0;
}
