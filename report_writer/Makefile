SERVICE_TITLE=Simple AI Fact Checker
PROJECT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PORT=8079
SERVICE_URL=http://localhost:${PORT}
SERVICE_ID=urn:ivcap:service:5a4f9c92-cbf9-5251-bd35-4568906405ba

run:
	poetry ivcap run -- --port ${PORT}

TEST_REQUEST=tests/solar.json
test-local:
	TOKEN=$(shell ivcap context get access-token --refresh-token); \
	curl -X POST --silent \
		-H "timeout: 300" \
		-H "Authorization: Bearer $$TOKEN" \
		-H "content-type: application/json" \
		--data @${TEST_REQUEST} http://localhost:${PORT}/ \
		| jq

test-job: IVCAP_API=https://develop.ivcap.net
test-job:
	TOKEN=$(shell ivcap context get access-token --refresh-token); \
	curl \
	-X POST \
	-H "content-type: application/json" \
	-H "Timeout: 20" \
	-H "Authorization: Bearer $$TOKEN" \
	--data @"test.json" \
	${IVCAP_API}/1/services2/${SERVICE_ID}/jobs?with-result-content=true | jq

docker-build:
	poetry ivcap docker-build

docker-run:
	docker run -it \
		-p ${PORT}:${PORT} \
		-e OPENAI_API_KEY=$(shell grep OPENAI_API_KEY .env | cut -d= -f2) \
		-e IVCAP_BASE_URL=$(shell ivcap context get url) \
		--platform=linux/arm64 \
		--rm \
		ivcap_simple_report_writer_agent_arm64:$(shell git rev-parse --short HEAD) \
		--port ${PORT}
