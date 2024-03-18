#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

// Fonction d'activation et sa d�riv�e
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double dSigmoid(double x) {
    return x * (1 - x);
}

// Fonction d'initialisation des poids
double init_weight() {
    return ((double)rand()) / ((double)RAND_MAX);
}

// Structure pour stocker les param�tres d'un r�seau
struct NetworkParameters {
    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];
    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];
    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];
    double training_inputs[numTrainingSets][numInputs];
    double training_outputs[numTrainingSets][numOutputs];
};

// Structure pour stocker les param�tres d'un thread
struct ThreadData {
    pthread_t thread;               // Identifiant du thread
    struct NetworkParameters* net;  // Param�tres du r�seau associ� au thread
};

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


// Fonction d'entra�nement pour un thread
void* trainNetwork(void* parameters) {
    struct ThreadData* threadData = (struct ThreadData*)parameters;
    struct NetworkParameters* params = threadData->net;


    int trainingSetOrder[] = {0,1,2,3};

    //Nombre d'�pochs pour le quel on entraine notre r�seau de neurones
    int numberOfEpochs = 10000;
     // Boucle d'entra�nement sur un certain nombre d'�pochs
    for (int epoch = 0; epoch < numberOfEpochs; ++epoch) {
        // Modifier l'ordre de notre ensemble d'entra�nement (� adapter selon vos besoins)
        shuffle(trainingSetOrder, numTrainingSets);

        // Cycle � travers chaque �l�ment de l'ensemble d'entra�nement
        for (int x = 0; x < numTrainingSets; ++x) {
            int i = trainingSetOrder[x];

            // Propagation avant
            // Calculez la fonction d'activation pour chaque neurone de la couche cach�e
            for (int j = 0; j < numHiddenNodes; ++j) {
                double activation = params->hiddenLayerBias[j];
                for (int k = 0; k < numInputs; ++k) {
                    activation += params->training_inputs[i][k] * params->hiddenWeights[k][j];
                }
                params->hiddenLayer[j] = sigmoid(activation);
            }

            // Calculez la fonction d'activation pour la couche de sortie
            for (int j = 0; j < numOutputs; ++j) {
                double activation = params->outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; ++k) {
                    activation += params->hiddenLayer[k] * params->outputWeights[k][j];
                }
                params->outputLayer[j] = sigmoid(activation);
            }

            // Backpropagation
            // Calculez l'erreur entre la sortie pr�dite et la sortie r�elle pour la couche de sortie
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; ++j) {
                double errorOutput = (params->training_outputs[i][j] - params->outputLayer[j]);
                deltaOutput[j] = errorOutput * dSigmoid(params->outputLayer[j]);
            }

            // Calculez l'erreur entre la sortie pr�dite et la sortie r�elle pour la couche cach�e
            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; ++j) {
                double errorHidden = 0.0;
                for (int k = 0; k < numOutputs; ++k) {
                    errorHidden += deltaOutput[k] * params->outputWeights[j][k];
                }
                deltaHidden[j] = errorHidden * dSigmoid(params->hiddenLayer[j]);
            }
const double lr = 0.1f;
            // Mise � jour des poids de sortie
            for (int j = 0; j < numOutputs; ++j) {
                params->outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; ++k) {
                    params->outputWeights[k][j] += params->hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            // Mise � jour des poids de la couche cach�e
            for (int j = 0; j < numHiddenNodes; ++j) {
                params->hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; ++k) {
                    params->hiddenWeights[k][j] += params->training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    return NULL;
}

int main(void) {
    // Initialisation des param�tres du r�seau 1
    struct NetworkParameters network1;
    for (int i = 0; i < numHiddenNodes; ++i) {
        for (int j = 0; j < numInputs; ++j) {
            network1.hiddenWeights[j][i] = init_weight();
        }
        network1.hiddenLayerBias[i] = init_weight();
    }
    for (int i = 0; i < numOutputs; ++i) {
        for (int j = 0; j < numHiddenNodes; ++j) {
            network1.outputWeights[j][i] = init_weight();
        }
        network1.outputLayerBias[i] = init_weight();
    }

    // Initialisation des param�tres du r�seau 2
    struct NetworkParameters network2;
    for (int i = 0; i < numHiddenNodes; ++i) {
        for (int j = 0; j < numInputs; ++j) {
            network2.hiddenWeights[j][i] = init_weight();
        }
        network2.hiddenLayerBias[i] = init_weight();
    }
    for (int i = 0; i < numOutputs; ++i) {
        for (int j = 0; j < numHiddenNodes; ++j) {
            network2.outputWeights[j][i] = init_weight();
        }
        network2.outputLayerBias[i] = init_weight();
    }

    // Cr�er une structure pour chaque r�seau et thread
    struct ThreadData threadData1, threadData2;  // Ajoutez autant de threads que n�cessaire

    // Allouer dynamiquement de la m�moire pour les param�tres du r�seau de chaque thread
    threadData1.net = malloc(sizeof(struct NetworkParameters));
    threadData2.net = malloc(sizeof(struct NetworkParameters));

    // Copier les param�tres du r�seau dans les structures de thread
    *threadData1.net = network1;
    *threadData2.net = network2;

    // Lancer les threads avec les structures de param�tres correspondantes
    pthread_create(&threadData1.thread, NULL, trainNetwork, (void*)&threadData1);
    pthread_create(&threadData2.thread, NULL, trainNetwork, (void*)&threadData2);

    // Attendre la fin de chaque thread
    pthread_join(threadData1.thread, NULL);
    pthread_join(threadData2.thread, NULL);

    // Lib�rer la m�moire
    free(threadData1.net);
    free(threadData2.net);



    return 0;
}
