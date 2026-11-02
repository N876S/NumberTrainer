#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

struct DataSet {
    std::vector<int> id;
    std::vector<std::vector<int>> data;
};

//28x28 data btw with the first character being label int
DataSet getData(const char* path){
    DataSet data;
    std::ifstream reader(path);

    if(!reader.is_open()){
        std::cout << "Failed to open file: " << path << "\n";
        return data;
    }

    std::string line;
    std::string term;

    //get rid of header line
    std::getline(reader, line);

    //READ LINE BY LINE
    while(std::getline(reader, line)){

        std::stringstream ss(line);
        std::vector<int> lineData;
        int counter = 0;

        while(std::getline(ss, term, ',')){
            if(counter == 0){
                data.id.push_back(std::stoi(term));
            } else if(counter <= 784){
                lineData.push_back(std::stoi(term));
            }
            counter++; 
        }

        if(!lineData.empty()){
            data.data.push_back(lineData);
        }
    }

    reader.close();

    return data;
}

class Neuron;
class Connection;

class Connection {
    public:

    Neuron* cn;
    float weight = 1.0f;
    float accWeightAdjust = 0.0f;

    Connection() : cn(nullptr), weight(0.0f) {}

    Connection(Neuron* cn, int numIncoming, int numOutgoing){
        //random bell curve
        static std::random_device rd;
        static std::mt19937 gen(rd());

        // Xavier / Glorot stddev formula for Sigmoid
        float stddev = std::sqrt(2.0f / (numIncoming + numOutgoing));
        std::normal_distribution<float> dist(0.0f, stddev);

        //generate random weight
        weight = dist(gen);

        this->cn = cn;
    }
};

class Neuron {
    public:

    int id;
    float value;
    std::vector<Connection> connections;
    std::vector<Neuron>* lastLayerNeurons;
    float bias = 0.0f;
    float accBiasAdjust = 0.0f;
    float accdCdA = 0.0f;

    Neuron() = default;

    Neuron(std::vector<Neuron>& lastLayerNeurons, int numOutgoing){
        bias = 0;
        connections.reserve(lastLayerNeurons.size());
        for(int i = 0; i < lastLayerNeurons.size(); i++){
            connections.push_back(Connection(&lastLayerNeurons[i], lastLayerNeurons.size(), numOutgoing));
        }
    }
};

class Layer {
    public:

    int id;
    std::vector<Neuron> neurons;
    std::vector<Neuron>* lastLayerNeurons;

    Layer(int numNeurons, int numOutgoing, std::vector<Neuron>* lastLayerNeurons = nullptr){
        neurons.reserve(numNeurons);
        for(int i = 0; i < numNeurons; i++){
            if(lastLayerNeurons != nullptr){
                neurons.push_back(Neuron(*lastLayerNeurons, numOutgoing));
            } else {
                neurons.push_back(Neuron());
            }
        }
    }
};

void calcAcc(Layer& layer, std::vector<float>& expectedNeuronValue, bool isOutputLayer){

    //get all layer weight adjustment amounts for one training example

    for(int i = 0; i < layer.neurons.size(); i++){

        float dCda = 0.0f;

        if(isOutputLayer == true){
            dCda = 2.0f*(layer.neurons[i].value - expectedNeuronValue[i]); //derivative of the cost function with respect to aj in this layer
        } else {
            dCda = layer.neurons[i].accdCdA;
        }

        //for each weight, run the cost calculation in the layer
        float dadz = layer.neurons[i].value * (1.0f - layer.neurons[i].value); //derivative of sigmoid (just z*(1-z) where z is already the value in the neuron)
        for(Connection& c : layer.neurons[i].connections){
            float dzdw = c.cn->value; //this is simply the neuron value the weight is connected to by the simple derivative

            float dCdw = dCda * dadz * dzdw; //cost for the weight for this layer in one training example

            //add to the weight accumulation
            c.accWeightAdjust += dCdw;

            //summing dCdA for hidden layers
            c.cn->accdCdA += dCda * dadz * c.weight;
        }

        //get all layer bias adjustment amounts for one training example

        float dCdb = dadz * dCda;
        layer.neurons[i].accBiasAdjust += dCdb;
    }
}

void calculateNetwork(std::vector<Layer>& network){
    for(int i = 1; i < network.size(); i++){
        for(Neuron& n : network[i].neurons){
            float x = 0.0f;

            //calculate z
            for(const Connection& c : n.connections){
                x += c.weight * (*(c.cn)).value;
            }
            x += n.bias;

            //sigmoid
            n.value = 1.0f/(1.0f + std::exp(-x));
        }
    }
}

int printFinalLayer(std::vector<Layer>& network){
    std::vector<float> results;

    std::cout << "\n";
    for(int i = 0; i < network[network.size()-1].neurons.size(); i++){
        std::cout << i << ": " << network[network.size()-1].neurons[i].value << "\n";
        results.push_back(network[network.size()-1].neurons[i].value);
    }
    auto i = std::max_element(results.begin(), results.end());
    int index = std::distance(results.begin(), i);

    std::cout << "Its a " << index << "!\n";

    return index;
}

void printMNISTImage(const std::vector<int>& imgData, int label) {
    std::cout << "\n--- Label: " << label << " ---\n";
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            int pixel = imgData[r * 28 + c]; // 0 to 255
            
            // Map pixel intensity to ASCII characters
            if (pixel > 150) {
                std::cout << "X "; // Dark (the digit)
            } else if (pixel > 40) {
                std::cout << ". "; // Medium gray (edges)
            } else {
                std::cout << "  "; // Light/Background (spaces)
            }
        }
        std::cout << "\n";
    }
    std::cout << "-------------------\n";
}

int main(){
    //load csv training data
    const char* trainPath = "../res/mnist_train.csv";
    const char* testPath = "../res/mnist_test.csv";  

    DataSet trainData = getData(trainPath);
    DataSet testData = getData(testPath);

    std::cout << "Train data processed\n";

    //create network
    std::vector<Layer> network;
    network.reserve(3);

    network.emplace_back(784, 512);
    network.emplace_back(512, 256);
    network.emplace_back(256, 128);
    network.emplace_back(128, 10);
    network.emplace_back(10, 1);

    //wire neuron connections
    for(int i = 1; i < network.size(); i++){
        int numIncoming = network[i-1].neurons.size();

        int numOutgoing = 1;
        if(i != network.size()-1){
            numOutgoing = network[i+1].neurons.size();
        }

        for(Neuron& n : network[i].neurons){
            n.connections.clear();
            for(Neuron& prevNeuron : network[i-1].neurons){
                n.connections.push_back(Connection(&prevNeuron, numIncoming, numOutgoing));
            }
        }
    }

    std::cout << "Network created successfully\n";

    //set inputs and test

    int epochCount = 10000;
    int batchSize = 32;
    float learningRate = 0.1f;
    float trainingExamples = 200.0f;

    float totalAccuracy = 0.0f;
    int totalCount = 0;

    std::vector<bool> lastHundredTrack;
    int lastHundredCount = 0;
    float lastHundredAccuracy = 0.0f;

    int streak = 0;
    int maxStreak = 0;

    for(int e = 0; e < epochCount; e++){
        for(int b = 0; b < batchSize; b++){
            for(int k = 0; k < trainingExamples; k++){
                //inputs the number data
                for(int i = 0; i < network[0].neurons.size(); i++){
                    network[0].neurons[i].value = (float)trainData.data[k+b*trainingExamples][i] / 255.0f;
                }
                
                calculateNetwork(network);

                //BACKPROPAGATION

                //reset dCdA for all neurons
                for(Layer& l : network){
                    for(Neuron& n : l.neurons){
                        n.accdCdA = 0.0f;
                    }
                }

                //getting a vector of expected result
                std::vector<float> expectedValues;
                for(int i = 0; i < 10; i++){
                    if(i == trainData.id[k+b*trainingExamples]){
                        expectedValues.push_back(1.0f);
                    } else {
                        expectedValues.push_back(0.0f);
                    }
                }

                //calculate for all layers
                for(int i = network.size()-1; i >= 1; i--){
                    if(i == network.size()-1){
                        calcAcc(network[i], expectedValues, true);
                    } else {
                        calcAcc(network[i], expectedValues, false);
                    }
                }
            }

            //divide and adjust weights/biases
            for(Layer& l : network){
                for(Neuron& n : l.neurons){
                    for(Connection& c : n.connections){
                        c.accWeightAdjust = c.accWeightAdjust / trainingExamples;
                        c.weight -= learningRate * c.accWeightAdjust;
                        c.accWeightAdjust = 0.0f;
                    }
                    n.accBiasAdjust = n.accBiasAdjust / trainingExamples;
                    n.bias -= learningRate * n.accBiasAdjust;
                    n.accBiasAdjust = 0.0f;
                }
            }
        }
        //----------------test a random one from test csv---------------------

        //generate random case to test
        std::random_device rd;  // Non-deterministic seed from the system
        std::mt19937 gen(rd()); // Standard Mersenne Twister engine
        std::uniform_int_distribution<int> distrib(0, testData.data.size()-1); // Range [0, 10000]

        int randomTestCase = distrib(gen);

        //inputs the test data
        for(int i = 0; i < network[0].neurons.size(); i++){
            network[0].neurons[i].value = (float)testData.data[randomTestCase][i] / 255.0f;
        }

        printMNISTImage(testData.data[randomTestCase], testData.id[randomTestCase]);
        
        calculateNetwork(network);

        int guess = printFinalLayer(network);
        std::cout << "Its actually a " << testData.id[randomTestCase] << "!\n";

        //track last 100
        if(lastHundredTrack.size() >= 100){
            lastHundredTrack.erase(lastHundredTrack.begin());
        }

        //if guess is correct
        if(guess == testData.id[randomTestCase]){
            totalCount++;
            streak++;
            lastHundredTrack.push_back(true);
        } else {
            streak = 0;
            lastHundredTrack.push_back(false);
        }

        //update max streak
        if(streak > maxStreak){
            maxStreak = streak;
        }

        //calculate last hundred accuracy
        lastHundredCount = 0;
        for(bool t : lastHundredTrack){
            if(t == true){
                lastHundredCount++;
            }
        }
        lastHundredAccuracy = (float) lastHundredCount / 100.0f;

        //calculate total accuracy
        totalAccuracy = (float) totalCount / (e+1);

        //display results
        std::cout << "Epoch: " << e+1 << " out of " << epochCount << "\n";
        std::cout << "Total Accuracy: " << totalAccuracy << "\n";
        std::cout << "Last 100 Accuracy: " << lastHundredAccuracy << "\n";
        std::cout << "Streak: x" << streak << "\n";
        std::cout << "Max Streak: x" << maxStreak << "\n";

    }
    return 0;
}