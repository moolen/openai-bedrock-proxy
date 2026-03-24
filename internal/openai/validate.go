package openai

func ValidateResponsesRequest(req ResponsesRequest) error {
	if req.Model == "" {
		return NewInvalidRequestError("model is required")
	}
	if isEmptyInput(req.Input) {
		return NewInvalidRequestError("input is required")
	}
	if !isSupportedInput(req.Input) {
		return NewInvalidRequestError("structured input is not supported")
	}
	if len(req.Tools) > 0 {
		return NewInvalidRequestError("tools are not supported")
	}
	if req.ToolChoice != nil {
		return NewInvalidRequestError("tool_choice is not supported")
	}
	if req.ParallelToolCalls != nil {
		return NewInvalidRequestError("parallel_tool_calls is not supported")
	}
	return nil
}

func isEmptyInput(input any) bool {
	if input == nil {
		return true
	}

	text, ok := input.(string)
	return ok && text == ""
}

func isSupportedInput(input any) bool {
	if input == nil {
		return false
	}
	_, ok := input.(string)
	return ok
}
