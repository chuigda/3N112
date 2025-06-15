package club.doki7.rkt.util;

import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;

public final class JULUtil {
    public static void setLogLevel(Level level) {
        Logger rootLogger = LogManager.getLogManager().getLogger("");
        rootLogger.setLevel(level);
        for (Handler h : rootLogger.getHandlers()) {
            h.setLevel(level);
        }
    }

    public static void addLogHandler(Handler handler) {
        Logger rootLogger = LogManager.getLogManager().getLogger("");
        rootLogger.addHandler(handler);
    }
}
