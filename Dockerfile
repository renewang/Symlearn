FROM python:3-onbuild
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
#EXPOSE 8888
# set the environment variable to avoid restricted 10G space through storage driver
ENV JOBLIB_TEMP_FOLDER "/usr/src/app/tmp" 
WORKDIR /usr/src/app/scripts
