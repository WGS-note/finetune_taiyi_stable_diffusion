docker stop jacktest
docker rm jacktest

prjPath=/data/renwanxin/txtdraw2022-develop

echo ""
echo ""
echo ""
echo "Test: curl 127.0.0.1:8001"
echo ""
echo ""
echo ""

docker run --rm --name jacktest \
	-v ${prjPath}:/data \
	-p 8001:6030 \
    -it jackoneflow:1.0 uwsgi --http 0.0.0.0:6030 --wsgi-file /data/hello.py


