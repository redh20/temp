oc new-app --as-deployment-config \
--context-dir=nodejs-helloworld \
https://github.com/${RHT_OCP4_GITHUB_USER}/DO180-apps#troubleshoot-s2i \
-i nodejs:12 --name nodejs-hello --build-env \
npm_config_registry=http://${RHT_OCP4_NEXUS_SERVER}/repository/npm-proxy
