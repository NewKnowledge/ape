node {
    docker.withRegistry('https://newknowledge.azurecr.io', 'acr-creds') {

        git url: "https://github.com/NewKnowledge/ape.git", credentialsId: '055e98d5-ce0c-45ef-bf0d-ddc6ed9b634a', branch: "${BRANCH_NAME}"

        sh "git rev-parse HEAD > .git/commit-id"
        def commit_id = readFile('.git/commit-id').trim()
        def clean_branchname = BRANCH_NAME.replaceAll("/", "-")
        println commit_id

        stage "build_docker_image"
        def api_image = docker.build("ape:${clean_branchname}", ".")

        stage "publish_docker_image"
        def images = [api_image]
        for (image in images) {
            image.push "${clean_branchname}"
            image.push "${commit_id}"
            if ("${BRANCH_NAME}" == "master") {
                image.push 'latest'
            }
        }

    //     /* Kick off another job to use the newly registered images */
    //     stage "Deploy on K8s"
    //     // Branches to deploy
    //     def buildableBranches = ["dev"]
    //     if (!buildableBranches.contains(BRANCH_NAME)) {
    //         currentBuild.result = 'ABORTED'
    //         error('Stopping early: Branch built is not listed to deploy to k8s.')
    //     }

    //     // Different configs for different branches
    //     def devMap  = [namespace: "quorum-dev", clusterId: "mario-monitor", releaseName: "ape-dev"]
    //     def stgMap  = [namespace: "quorum-stg", clusterId: "mario-monitor", releaseName: "ape-stg"]
    //     def prodMap  = [namespace: "quorum-prod", clusterId: "prod", releaseName: "ape-prod"]

    //     // Select "active" map
    //     def activeMap = [:]
    //     if ("${BRANCH_NAME}" == "master") {
    //         activeMap = prodMap
    //     } else if ("${BRANCH_NAME}" == "stg") {
    //         activeMap = stgMap
    //     } else {
    //         activeMap = devMap
    //     }
    //     // Run k8s job with config
    //     build job: "deploy-k8", parameters: [
    //         string(name: "imageTag", value: "${commit_id}"),
    //         string(name: "chartName", value: "nk.external-service"),
    //         string(name: "valueFilename", value: "ape.yaml"),
    //         string(name: "namespace", value: "${activeMap.namespace}"),
    //         string(name: "clusterId", value: "${activeMap.clusterId}"),
    //         string(name: "releaseName", value: "${activeMap.releaseName}")
    //     ]
    }
}
