node {
    stage('checkout') {
        mail('is starting')
        try{
        git credentialsId: '1cca939a-c7d2-449c-840c-6e97027515d3', url: 'https://github.com/baddwolff/Interview'
        }
        catch(err) {
            mail("has error while cloning...")
            throw err
        }
    }

    stage('check java') {
        sh "java -version"
    }

    stage('clean') {
        try{
        sh "./gradle clean --no-daemon"
        }
        catch(err){
            mail("has error while cleaning...")
            throw err
        }
        
    }
    
    stage('npm install') {
        try{
            sh "./gradle npm_install -PnodeInstall --no-daemon"
        }catch(err){
            mail("has error while installin node modules...")
            throw err
        }
    }
    
    stage('backend tests') {
        input 'Continue to Testing?'
        try {
            // sh "./gradle test integrationTest -PnodeInstall --no-daemon"
            sh "./gradle test  -PnodeInstall --no-daemon"
        } catch(err) {
            mail("has failed during in the backend tests...")
            throw err
        } finally {
            junit '**/build/**/TEST-*.xml'
        }
    }

    stage('frontend tests') {
        try {
            sh "./gradle npm_run_test -PnodeInstall --no-daemon"
        } catch(err) {
            mail("has failed during the front end tests...")
            throw err
        }
    }
    
    stage('packaging') {
        try{
            sh "./gradle bootJar -x test -Pprod -PnodeInstall --no-daemon"
            archiveArtifacts artifacts: '**/build/libs/*.war', fingerprint: true
            
            // sh "./gradle build"
            // archiveArtifacts artifacts: 'build/libs/*.war', fingerprint: true
         
        }
        catch(err){
            mail("has failed while packaging...")
            throw err
        }
    }
    mail("is waiting for deployment...")
    
    stage('deploying') {
        input 'Continue to Deployment?'
        try{
            sh "echo DEPLOYING..."
            dir('build/libs') {
                // some block
                sh 'java -jar interview-0.0.1-SNAPSHOT.war'
            }
            sh "echo DONE."
            mail("has been succesfully deployed!")
        }
        catch(err){
            mail("deployement has failed!")
        }
    }

}
def mail(status){
    emailext body: '''$JOB_NAME - Build#$BUILD_NUMBER Current Status: $BUILD_STATUS.<br/>
<br/>
Check console <a href="${env.BUILD_URL}">output</a> to view full results.<br/>
If you cannot connect to the build server, check the attached logs.<br/>
<br/>
--<br/>
Following is the last 100 lines of the log.<br/>
<br/>
--LOG-BEGIN--<br/>
<pre style='line-height: 22px; display: block; color: #333; font-family: Monaco,Menlo,Consolas,"Courier New",monospace; padding: 10.5px; margin: 0 0 11px; font-size: 13px; word-break: break-all; word-wrap: break-word; white-space: pre-wrap; background-color: #f5f5f5; border: 1px solid #ccc; border: 1px solid rgba(0,0,0,.15); -webkit-border-radius: 4px; -moz-border-radius: 4px; border-radius: 4px;'>
${BUILD_LOG, maxLines=100, escapeHtml=true}
</pre>
--LOG-END--''', 
    subject: "Jenkins/${env.JOB_NAME}/Build#${env.BUILD_NUMBER} ${status}", 
    to: 'adityagirisha@gmail.com'
}