SERVICE_TITLE=Simple AI Fact Checker
PROJECT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PORT=8078
SERVICE_URL=http://localhost:${PORT}
SERVICE_ID=urn:ivcap:service:5a4f9c92-cbf9-5251-bd35-4568906405ba

run:
	poetry ivcap run -- --port ${PORT}

TEST_REQUEST=${PROJECT_DIR}/tests/space_com.json
test-local:
	curl -i -X POST -H "timeout: 300" -H "content-type: application/json" --data @${TEST_REQUEST} http://localhost:${PORT}/

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
		--platform=linux/arm64 \
		--rm \
		ivcap_simple_fact_checker_agent_arm64:latest \
		--port ${PORT}
