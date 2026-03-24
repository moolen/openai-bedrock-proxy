package openai

import "errors"

type ErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type ErrorResponse struct {
	Error ErrorBody `json:"error"`
}

type InvalidRequestError struct {
	Message string
}

func NewInvalidRequestError(message string) InvalidRequestError {
	return InvalidRequestError{Message: message}
}

func (e InvalidRequestError) Error() string {
	return e.Message
}

func ErrorResponseFrom(err error) ErrorResponse {
	if err == nil {
		return ErrorResponse{
			Error: ErrorBody{
				Message: "internal server error",
				Type:    "server_error",
			},
		}
	}

	var invalidRequest InvalidRequestError
	if errors.As(err, &invalidRequest) {
		return ErrorResponse{
			Error: ErrorBody{
				Message: invalidRequest.Message,
				Type:    "invalid_request_error",
			},
		}
	}

	return ErrorResponse{
		Error: ErrorBody{
			Message: err.Error(),
			Type:    "server_error",
		},
	}
}
