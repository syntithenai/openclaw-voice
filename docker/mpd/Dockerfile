FROM alpine:3.19

# Install MPD and required packages
RUN apk add --no-cache \
    mpd \
    mpc \
    bash \
    pulseaudio \
    && rm -rf /var/cache/apk/*

# Create necessary directories
RUN mkdir -p /var/lib/mpd/playlists \
    /var/log/mpd \
    /run/mpd \
    /music

# Copy configuration
COPY mpd.conf /etc/mpd.conf

# Set proper permissions
RUN chown -R mpd:audio /var/lib/mpd /var/log/mpd /run/mpd

# Expose MPD port
EXPOSE 6600

# Run MPD as mpd user
USER mpd

# Start MPD in foreground
CMD ["mpd", "--no-daemon", "/etc/mpd.conf"]
