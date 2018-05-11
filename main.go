package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"github.com/Pattern-Recognition4-5/neuralNetwork"
	//"github.com/Pattern-Recognition4-5/cnn"
	"github.com/sirupsen/logrus"
	"io"
	"os"
)

const (
	pixelRange = 255
)

func main() {
	nn()
}

func nn() {
	fmt.Println("Training Neural Network......")

	//Note:  Below code is referenced from PA3 with modification for Neural Network

	//load data files (to be taken from console)
	//todo: change flag names
	testLabelFile := flag.String("tl", "", "test label file")
	testImageFile := flag.String("ti", "", "test image file")
	sourceLabelFile := flag.String("sl", "", "source label file")
	sourceImageFile := flag.String("si", "", "source image file")
	flag.Parse()

	fmt.Println(sourceLabelFile , sourceImageFile)
	logrus.Info("Loading MNIST training data set ......")
	//
	labelData := ReadLabels(OpenFile(*sourceLabelFile))
	imageData, width, height := ReadImages(OpenFile(*sourceImageFile)) // training file

	//labelData := ReadLabels(OpenFile(*testLabelFile))
	//imageData, width, height := ReadImages(OpenFile(*testImageFile)) // testing  file

	//Not Required
	fmt.Println("Total Source Images : ", len(imageData), len(imageData[0]), width, height)
	fmt.Println("Label : ", len(labelData), labelData[0:10])

	inputs := makeXCoordinates(imageData)
	targets := makeYCoordinates(labelData) //todo: put testData for getting testing error

	//create neural network with
	/*
		part b : hidden nodes = 30, outputnodes = 10, learning rate η=0.1, number of images = 1000 (i.e all training images)
		iterations = 30 (nn.Train)

		Plot
		1. [DONE] training error (Mean squared normalized error ?)
		2. [DONE] testing error
		3. [DONE] criterion function on training data set
		4. [DONE] criterion function on testing data set
		5. the learning speed of the hidden layer (the average absolute changes of weights divided by the values of the weights).
	*/
	//The extraction routines reshape (so as that each digit is represented by a 1-D column vector of size 784)
	//nn := neuralNetwork.NewNetwork(784, 100, 10, false, 0.25, 0.1)
	nn := neuralNetwork.NewNetwork(784, 30, 10, false, 5, 0.1)



	//specify number of iterations
	nn.Train(inputs, targets, 30) //20 iterations


	logrus.Info(">>>>>>>>",nn.OutputLayer) //Criterion Function

	//Load test data
	var testLabelData []byte
	var testImageData [][]byte
	if *testLabelFile != "" && *testImageFile != "" {
		logrus.Println("Loading test data...")
		testLabelData = ReadLabels(OpenFile(*testLabelFile))
		testImageData, _, _ = ReadImages(OpenFile(*testImageFile))
	}

	test_inputs := makeXCoordinates(testImageData)
	test_targets := makeYCoordinates(testLabelData)

	correct_ct := 0
	for i, p := range test_inputs {
		//fmt.Println(nn.Forward(p))
		y := argmaxFunction(nn.Forward(p))
		yy := argmaxFunction(test_targets[i])
		//fmt.Println(y,yy)
		if y == yy {
			correct_ct += 1
		}
	}

	fmt.Println("correct rate: ", float64(correct_ct)/float64(len(test_inputs)), correct_ct, len(test_inputs))
}



func ReadLabels(r io.Reader) (labels []byte) {
	header := [2]int32{}
	binary.Read(r, binary.BigEndian, &header)
	labels = make([]byte, header[1])
	r.Read(labels)
	return
}

func ReadImages(r io.Reader) (images [][]byte, width, height int) {
	header := [4]int32{}
	binary.Read(r, binary.BigEndian, &header)
	images = make([][]byte, header[1])
	width, height = int(header[2]), int(header[3])
	for i := 0; i < len(images); i++ {
		images[i] = make([]byte, width*height)
		r.Read(images[i])
	}
	return
}

func makeXCoordinates(M [][]byte) [][]float64 {
	rows := len(M)
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, len(M[i]))
		for j := 0; j < len(M[i]); j++ {
			result[i][j] = getWeightPixel(M[i][j])
		}
	}
	return result
}

func makeYCoordinates(N []byte) [][]float64 {
	result := make([][]float64, len(N))
	for i := 0; i < len(result); i++ {
		tmp := make([]float64, 10)
		for j := 0; j < 10; j++ {
			tmp[j] = 0.1
		}
		tmp[N[i]] = 0.9
		result[i] = tmp
	}
	return result
}

func OpenFile(path string) *os.File {
	file, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
	return file
}

func getWeightPixel(px byte) float64 {
	return float64(px)/pixelRange*0.9 + 0.1
}

func argmaxFunction(A []float64) int {
	x := 0
	v := -1.0
	for i, a := range A {
		if a > v {
			x = i
			v = a
		}
	}
	return x
}
