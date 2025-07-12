package club.doki7.rkt.launch.nn;

import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.FloatPtr;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.vulkan.command.VulkanLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;

public class MNIST_Swing {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main(String[] ignored) {
        LinkedBlockingQueue<boolean[][]> inputQueue = new LinkedBlockingQueue<>();

        JLabel[] numPossibilities = new JLabel[10];
        for (int i = 0; i < numPossibilities.length; i++) {
            numPossibilities[i] = new JLabel("数字 " + i + ": 0%");
            numPossibilities[i].setBorder(BorderFactory.createEmptyBorder(2, 2, 2, 2));
        }

        new Thread(() -> {
            Buffer.OptionsInit inputBufferOptionsInit = new Buffer.OptionsInit();
            inputBufferOptionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER);
            inputBufferOptionsInit.mapped = true;
            inputBufferOptionsInit.coherent = true;
            Buffer.Options inputBufferOptions = inputBufferOptionsInit.build();

            try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
                 ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
                 RenderContext cx = RenderContext.createHeadless(libVulkan, libVMA, new RenderConfig());
                 MLPFactory mlpFactory = new MLPFactory(cx);
                 MLP mlp = buildMLP(mlpFactory);
                 Buffer inputBuffer = Buffer.create(
                         cx,
                         MNIST_IMAGE_SIZE * Float.BYTES,
                         false,
                         inputBufferOptions
                 );
                 MLPInferTask inferTask = new MLPInferTask(mlp, 1, inputBuffer, true, true)) {
                FloatPtr inputBufferMapped =
                        Objects.requireNonNull(FloatPtr.checked(inferTask.inputBuffer.mapped));
                FloatPtr outputBufferMapped =
                        Objects.requireNonNull(FloatPtr.checked(inferTask.outputBufferList.getLast().mapped));

                while (true) {
                    boolean[][] input = inputQueue.take();

                    for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
                        int row = i / 28;
                        int col = i % 28;
                        inputBufferMapped.write(i, input[row][col] ? 1.0f : 0.0f);
                    }

                    inferTask.executeBatch(0);

                    float[] exp = new float[10];
                    float maxExp = Float.MIN_VALUE;
                    for (int i = 0; i < exp.length; i++) {
                        exp[i] = outputBufferMapped.read(i);
                        if (exp[i] > maxExp) {
                            maxExp = exp[i];
                        }
                    }

                    float sum = 0.0f;
                    for (int i = 0; i < exp.length; i++) {
                        exp[i] = (float) Math.exp(exp[i] - maxExp);
                        sum += exp[i];
                    }

                    final float sum1 = sum;
                    SwingUtilities.invokeLater(() -> {
                        for (int i = 0; i < exp.length; i++) {
                            float percentage = (exp[i] / sum1) * 100.0f;
                            numPossibilities[i].setText(String.format("数字 %d: %.2f%%", i, percentage));
                        }
                    });
                }
            } catch (Throwable e) {
                e.printStackTrace(System.err);
            }
        }).start();

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("MNIST 手写数字识别");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            DrawingPanel drawingPanel = new DrawingPanel() {
                @Override
                public void onCanvasUpdate() {
                    boolean[][] input = new boolean[28][28];
                    for (int row = 0; row < 28; row++) {
                        System.arraycopy(grid[row], 0, input[row], 0, 28);
                    }
                    inputQueue.offer(input);
                }
            };

            JPanel controlPanel = new JPanel();
            controlPanel.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
            BoxLayout layout = new BoxLayout(controlPanel, BoxLayout.Y_AXIS);
            controlPanel.setLayout(layout);
            JButton clearButton = new JButton("清除");
            clearButton.setPreferredSize(new Dimension(100, 30));
            clearButton.addActionListener(_ -> {
                drawingPanel.clearCanvas();
                for (int i = 0; i < numPossibilities.length; i++) {
                    numPossibilities[i].setText("数字 " + i + ": 0%");
                }
            });
            for (JLabel label : numPossibilities) {
                controlPanel.add(label);
            }
            controlPanel.add(clearButton);

            frame.add(drawingPanel, BorderLayout.CENTER);
            frame.add(controlPanel, BorderLayout.EAST);

            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setResizable(false);
            frame.setVisible(true);
        });
    }

    private static MLP buildMLP(MLPFactory factory) throws IOException, RenderException {
        MLPOptions options = new MLPOptions(
                MNIST_IMAGE_SIZE,
                Arrays.asList(
                        new MLPOptions.Layer(300, Activation.RELU, 32),
                        new MLPOptions.Layer(100, Activation.RELU, 32),
                        new MLPOptions.Layer(10, Activation.LINEAR, 2)
                ),
                true
        );
        MLP mlp = factory.createModel(options);

        List<MemorySegment> weightList = new ArrayList<>();
        List<MemorySegment> biasList = new ArrayList<>();
        for (int i = 0; i < weightFileNameList.size(); i++) {
            String weightFileName = weightFileNameList.get(i);
            String biasFileName = biasFileNameList.get(i);

            byte[] weights = Files.readAllBytes(Path.of("resc", "nn", weightFileName));
            byte[] biases = Files.readAllBytes(Path.of("resc", "nn", biasFileName));
            assert weights.length == weightFileSize.get(i) && biases.length == biasFileSize.get(i);

            weightList.add(MemorySegment.ofArray(weights));
            biasList.add(MemorySegment.ofArray(biases));
        }

        mlp.uploadWeights(weightList, biasList);
        return mlp;
    }

    private static final java.util.List<String> weightFileNameList = java.util.List.of(
            "weights_L1_784x300.bin",
            "weights_L2_300x100.bin",
            "weights_L3_100x10.bin"
    );
    private static final java.util.List<Long> weightFileSize = java.util.List.of(
            784L * 300 * Float.BYTES,
            300L * 100 * Float.BYTES,
            100L * 10 * Float.BYTES
    );
    private static final java.util.List<String> biasFileNameList = java.util.List.of(
            "biases_L1_784x300.bin",
            "biases_L2_300x100.bin",
            "biases_L3_100x10.bin"
    );
    private static final java.util.List<Long> biasFileSize = List.of(
            300L * Float.BYTES,
            100L * Float.BYTES,
            10L * Float.BYTES
    );
    private static final int MNIST_IMAGE_SIZE = 28 * 28;
}

abstract class DrawingPanel extends JPanel {
    private static final int GRID_SIZE = 28;
    private static final int CELL_SIZE = 20;
    private static final int PANEL_WIDTH = GRID_SIZE * CELL_SIZE;
    private static final int PANEL_HEIGHT = GRID_SIZE * CELL_SIZE;

    protected final boolean[][] grid;

    public DrawingPanel() {
        this.grid = new boolean[GRID_SIZE][GRID_SIZE];

        setPreferredSize(new Dimension(PANEL_WIDTH, PANEL_HEIGHT));
        setBackground(Color.WHITE);

        MouseAdapter mouseAdapter = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                drawOnGrid(e);
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                drawOnGrid(e);
            }
        };

        addMouseListener(mouseAdapter);
        addMouseMotionListener(mouseAdapter);
    }

    private void drawOnGrid(MouseEvent e) {
        int x = e.getX();
        int y = e.getY();

        int col = x / CELL_SIZE;
        int row = y / CELL_SIZE;

        if (row >= 0 && row < GRID_SIZE && col >= 0 && col < GRID_SIZE) {
            boolean previous = grid[row][col];
            grid[row][col] = true;
            repaint();

            if (!previous) {
                onCanvasUpdate();
            }
        }
    }

    public void clearCanvas() {
        for (boolean[] row : grid) {
            Arrays.fill(row, false);
        }
        repaint();
    }

    public abstract void onCanvasUpdate();

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        g.setColor(Color.BLACK);
        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                if (grid[row][col]) {
                    g.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }
        }

        g.setColor(Color.LIGHT_GRAY);
        for (int i = 0; i <= GRID_SIZE; i++) {
            g.drawLine(i * CELL_SIZE, 0, i * CELL_SIZE, PANEL_HEIGHT);
        }
        for (int i = 0; i <= GRID_SIZE; i++) {
            g.drawLine(0, i * CELL_SIZE, PANEL_WIDTH, i * CELL_SIZE);
        }
    }
}
