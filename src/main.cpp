#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

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
    float weight;

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
    float bias;

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

void calculateNetwork(std::vector<Layer>& network){
    for(int i = 1; i < network.size(); i++){
        for(Neuron& n : network[i].neurons){
            float x = 0.0f;

            for(Connection c : n.connections){
                x += c.weight * (*(c.cn)).value;
            }
            x += n.bias;

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

int main(){
    const char* trainPath = "../res/mnist_train.csv";
    const char* testPath = "../res/mnist_test.csv";  

    DataSet trainData = getData(trainPath);

    std::cout << "Train data processed\n";

    std::vector<Layer> network;
    network.reserve(3);

    network.emplace_back(784, 300);
    network.emplace_back(300, 10);
    network.emplace_back(10, 1);

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

    float accuracy = 0.0f;
    int timesRight = 0;

    for(int k = 0; k < 10000; k++){
        for(int i = 0; i < network[0].neurons.size(); i++){
            network[0].neurons[i].value = (float)trainData.data[k][i] / 255.0f;
        }

        std::cout << "Network inputs processed\n";
        
        calculateNetwork(network);

        std::cout << "Network processed\n";

        int guess = printFinalLayer(network);
        std::cout << "Its actually a " << trainData.id[k] << "!\n";

        if(guess == trainData.id[k]){
            timesRight++;
        }
        accuracy = (float) timesRight / k;
        std::cout << "Accuracy: " << accuracy << "\n";
    }

    return 0;
}