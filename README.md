#Build:

docker build -t taxid_server .

#Run:

docker run -p 8080:8080 taxid_server

#Editing the swagger file:

In case the swagger file needs to be edited (for example, the host URL), make sure that the swagger_server/swagger/swagger.yaml is up to date with those changes as well.