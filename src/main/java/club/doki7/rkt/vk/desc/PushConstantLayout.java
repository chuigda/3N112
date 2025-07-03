package club.doki7.rkt.vk.desc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class PushConstantLayout {
    public final List<PushConstantRange> ranges;
    public final int totalSize;
    public final List<Integer> offsets;

    public PushConstantLayout(List<PushConstantRange> ranges) {
        int totalSize = 0;
        List<Integer> offsets = new ArrayList<>();
        for (PushConstantRange range : ranges) {
            offsets.add(totalSize);
            totalSize += range.size;
        }

        this(
                Collections.unmodifiableList(ranges),
                totalSize,
                Collections.unmodifiableList(offsets)
        );
    }

    private PushConstantLayout(
            List<PushConstantRange> ranges,
            int totalSize,
            List<Integer> offsets
    ) {
        this.ranges = ranges;
        this.totalSize = totalSize;
        this.offsets = offsets;
    }
}
