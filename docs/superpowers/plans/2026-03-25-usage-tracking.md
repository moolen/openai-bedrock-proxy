# Usage Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-memory token usage tracking with overall and per-session inspection endpoints.

**Architecture:** Normalize Bedrock token usage into internal response structs, record it through a small in-memory tracker, and expose totals through HTTP handlers. Session tracking uses `previous_response_id` chains for `/v1/responses` and `X-Session-ID` for explicit grouping.

**Tech Stack:** Go, net/http, AWS Bedrock SDK, existing in-memory stores, table-style unit tests

---

### Task 1: Define failing usage tests

**Files:**
- Modify: `internal/proxy/chat_service_test.go`
- Modify: `internal/proxy/service_test.go`
- Modify: `internal/httpserver/server_test.go`
- Modify: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write failing tests for non-stream chat usage**
- [ ] **Step 2: Run targeted tests and confirm they fail for missing usage**
- [ ] **Step 3: Write failing tests for tracker aggregation and usage endpoints**
- [ ] **Step 4: Run targeted tests and confirm they fail for missing tracking**

### Task 2: Normalize Bedrock usage

**Files:**
- Modify: `internal/bedrock/translate_response.go`
- Modify: `internal/bedrock/chat_translate.go`
- Modify: `internal/bedrock/chat_stream.go`
- Modify: `internal/bedrock/client.go`

- [ ] **Step 1: Add internal usage fields to Bedrock response structs**
- [ ] **Step 2: Capture Bedrock non-stream usage from `ConverseOutput`**
- [ ] **Step 3: Capture streaming usage metadata for responses and chat**
- [ ] **Step 4: Populate non-stream chat completion `usage`**

### Task 3: Add in-memory tracker

**Files:**
- Create: `internal/usage/tracker.go`
- Create: `internal/openai/usage_types.go`
- Modify: `internal/conversation/types.go`
- Modify: `internal/conversation/store.go`

- [ ] **Step 1: Add tracker types and aggregation logic**
- [ ] **Step 2: Extend conversation records with session ids**
- [ ] **Step 3: Add JSON response types for usage endpoints**

### Task 4: Wire services and handlers

**Files:**
- Modify: `internal/proxy/service.go`
- Modify: `internal/httpserver/server.go`
- Modify: `cmd/openai-bedrock-proxy/main.go`
- Modify: `README.md`

- [ ] **Step 1: Record usage after successful requests**
- [ ] **Step 2: Resolve session ids from `X-Session-ID` and stored continuations**
- [ ] **Step 3: Add `GET /v1/usage` and `GET /v1/usage/{session_id}`**
- [ ] **Step 4: Document how to measure usage**

### Task 5: Verify

**Files:**
- Modify: `internal/*/*_test.go`

- [ ] **Step 1: Run targeted package tests**
- [ ] **Step 2: Fix regressions**
- [ ] **Step 3: Re-run focused tests until green**
