package machinelearningmodels

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// ReadCSV reads a CSV file and returns its content as a slice of string slices.
func ReadCSV(filePath string) ([][]string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening file: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error reading CSV: %w", err)
	}

	return records, nil
}

// StringToFloat64 converts a string slice to a float64 slice.
func StringToFloat64(data []string) ([]float64, error) {
	result := make([]float64, len(data))
	for i, v := range data {
		val, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, fmt.Errorf("error converting string to float64: %w", err)
		}
		result[i] = val
	}
	return result, nil
}

// CalculateMean calculates the mean of a slice of float64.
func CalculateMean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// CalculateStandardDeviation calculates the standard deviation of a slice of float64.
func CalculateStandardDeviation(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	mean := CalculateMean(data)
	sumOfSquares := 0.0
	for _, v := range data {
		sumOfSquares += math.Pow(v-mean, 2)
	}
	variance := sumOfSquares / float64(len(data))
	return math.Sqrt(variance)
}

// NormalizeData normalizes a slice of float64 to a range of 0 to 1.
func NormalizeData(data []float64) []float64 {
	if len(data) == 0 {
		return []float64{}
	}

	min := data[0]
	max := data[0]

	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	if min == max {
		// All values are the same, return a slice of 0s.
		normalized := make([]float64, len(data))
		return normalized
	}

	normalized := make([]float64, len(data))
	for i, v := range data {
		normalized[i] = (v - min) / (max - min)
	}
	return normalized
}

// SplitData splits data into training and testing sets.
func SplitData(data [][]string, trainRatio float64) ([][]string, [][]string) {
	trainSize := int(float64(len(data)) * trainRatio)
	return data[:trainSize], data[trainSize:]
}

// ConvertStringSliceToInterfaceSlice converts a string slice to an interface slice.
func ConvertStringSliceToInterfaceSlice(data []string) []interface{} {
	interfaceSlice := make([]interface{}, len(data))
	for i, d := range data {
		interfaceSlice[i] = d
	}
	return interfaceSlice
}

// InterfaceSliceToStringSlice converts an interface slice to a string slice.
func InterfaceSliceToStringSlice(data []interface{}) []string {
	stringSlice := make([]string, len(data))
	for i, d := range data {
		stringSlice[i] = fmt.Sprint(d)
	}
	return stringSlice
}

// WriteCSV writes data to a CSV file.
func WriteCSV(filePath string, data [][]string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, record := range data {
		err := writer.Write(record)
		if err != nil {
			return fmt.Errorf("failed to write record: %w", err)
		}
	}
	return nil
}

// ValidateFilePath checks if a file path is valid (exists and is a file).
func ValidateFilePath(filePath string) error {
	fileInfo, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		return fmt.Errorf("file does not exist: %s", filePath)
	}
	if err != nil {
		return fmt.Errorf("error checking file: %w", err)
	}
	if fileInfo.IsDir() {
		return fmt.Errorf("path is a directory, not a file: %s", filePath)
	}
	return nil
}

// ParseStringSliceToFloat64OrString converts a string slice to a slice of either float64 or string, based on whether the string can be parsed to a float64.
func ParseStringSliceToFloat64OrString(data []string) []interface{} {
	result := make([]interface{}, len(data))
	for i, v := range data {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			result[i] = f
		} else {
			result[i] = v
		}
	}
	return result
}

// RemoveEmptyLines removes empty lines from CSV data.
func RemoveEmptyLines(data [][]string) [][]string {
	result := make([][]string, 0)
	for _, row := range data {
		isEmpty := true
		for _, cell := range row {
			if strings.TrimSpace(cell) != "" {
				isEmpty = false
				break
			}
		}
		if !isEmpty {
			result = append(result, row)
		}
	}
	return result
}

// IsHeaderRow checks if a row in a CSV is a header row by checking if all cells are strings (unable to convert to float64).
func IsHeaderRow(row []string) bool {
	for _, cell := range row {
		if _, err := strconv.ParseFloat(cell, 64); err == nil {
			return false // At least one cell can be converted to float64
		}
	}
	return true // All cells are strings
}

// ColumnExists checks if a column with the given header exists in the CSV data.
func ColumnExists(headerRow []string, columnName string) bool {
	for _, header := range headerRow {
		if header == columnName {
			return true
		}
	}
	return false
}

// GetColumnIndex returns the index of a column with the given header in the CSV data. Returns -1 if the column does not exist.
func GetColumnIndex(headerRow []string, columnName string) int {
	for i, header := range headerRow {
		if header == columnName {
			return i
		}
	}
	return -1
}

// ReadCSVWithHeader reads a CSV file and returns its content as a slice of string slices, along with the header row.
func ReadCSVWithHeader(filePath string) ([][]string, []string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("error opening file: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)

	// Read the header row
	headerRow, err := reader.Read()
	if err != nil {
		if err == io.EOF {
			return nil, nil, fmt.Errorf("empty CSV file")
		}
		return nil, nil, fmt.Errorf("error reading header row: %w", err)
	}

	// Read the remaining records
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("error reading CSV: %w", err)
	}

	return records, headerRow, nil
}