package openai

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
	switch typed := err.(type) {
	case InvalidRequestError:
		return ErrorResponse{
			Error: ErrorBody{
				Message: typed.Message,
				Type:    "invalid_request_error",
			},
		}
	default:
		return ErrorResponse{
			Error: ErrorBody{
				Message: err.Error(),
				Type:    "server_error",
			},
		}
	}
}
