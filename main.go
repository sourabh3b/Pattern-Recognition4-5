package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"github.com/Pattern-Recognition4-5/neuralNetwork"
	"github.com/sirupsen/logrus"
	"io"
	"os"
)

const (
	pixelRange = 255
)

func main() {
	fmt.Println("Training Neural Network......")

	//Note:  Below code is referenced from PA3 with modification for Neural Network

	//load data files (to be taken from console)
	//todo: change flag names
	testLabelFile := flag.String("tl", "", "test label file")
	testImageFile := flag.String("ti", "", "test image file")
	sourceLabelFile := flag.String("sl", "", "source label file")
	sourceImageFile := flag.String("si", "", "source image file")
	flag.Parse()

	logrus.Info("Loading MNIST training data set ......")

	labelData := ReadMNISTLabels(OpenFile(*sourceLabelFile))
	imageData, width, height := ReadMNISTImages(OpenFile(*sourceImageFile))

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
		1. training error (Mean squared normalized error ?)
		2. testing error
		3. criterion function on training data set
		4. criterion function on testing data set
		5. the learning speed of the hidden layer (the average absolute changes of weights divided by the values of the weights).
	*/
	nn := neuralNetwork.NewNetwork(784, 100, 10, false, 0.25, 0.1)

	fmt.Println("Err Output : ",nn.ErrOutput)
	//specify number of iterations
	nn.Train(inputs, targets, 10) //20 iterations

	//Load test data
	var testLabelData []byte
	var testImageData [][]byte
	if *testLabelFile != "" && *testImageFile != "" {
		logrus.Println("Loading test data...")
		testLabelData = ReadMNISTLabels(OpenFile(*testLabelFile))
		testImageData, _, _ = ReadMNISTImages(OpenFile(*testImageFile))
	}

	test_inputs := makeXCoordinates(testImageData)
	test_targets := makeYCoordinates(testLabelData)

	correct_ct := 0
	for i, p := range test_inputs {
		//fmt.Println(nn.Forward(p))
		y := argmax(nn.Forward(p))
		yy := argmax(test_targets[i])
		//fmt.Println(y,yy)
		if y == yy {
			correct_ct += 1
		}
	}

	fmt.Println("correct rate: ", float64(correct_ct)/float64(len(test_inputs)), correct_ct, len(test_inputs))
}

func ReadMNISTLabels(r io.Reader) (labels []byte) {
	header := [2]int32{}
	binary.Read(r, binary.BigEndian, &header)
	labels = make([]byte, header[1])
	r.Read(labels)
	return
}

func ReadMNISTImages(r io.Reader) (images [][]byte, width, height int) {
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
			result[i][j] = pixelWeight(M[i][j])
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

func pixelWeight(px byte) float64 {
	return float64(px)/pixelRange*0.9 + 0.1
}

func argmax(A []float64) int {
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
