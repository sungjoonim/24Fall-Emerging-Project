package com.example.pj4combined

import android.Manifest.permission.CAMERA
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.pj4combined.cameraView.CameraScreenWithLatencyLogging
import com.example.pj4combined.chatView.ChatRoute
import com.example.pj4combined.chatView.LoadingRoute
import com.example.pj4combined.ui.theme.PJ4COMBINEDTheme

const val START_SCREEN = "start_screen"
const val CHAT_SCREEN = "chat_screen"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val startPreview = remember { mutableStateOf(false) }
            val startLLM = remember { mutableStateOf(false) }

            PJ4COMBINEDTheme {
                Scaffold(topBar = { AppBar() })
                { innerPadding ->
                    // A surface container using the 'background' color from the theme
                    Column (modifier = Modifier.fillMaxSize().padding(innerPadding)){
                        Box (modifier = Modifier.fillMaxWidth().fillMaxHeight(0.7F)){
                            if (startLLM.value) {
                                Surface(modifier = Modifier.fillMaxSize(),color = MaterialTheme.colorScheme.background) {
                                    val navController = rememberNavController()

                                    NavHost(navController = navController, startDestination = START_SCREEN) {
                                        composable(START_SCREEN) {
                                            LoadingRoute(
                                                onModelLoaded = {
                                                    navController.navigate(CHAT_SCREEN) {
                                                        popUpTo(START_SCREEN) { inclusive = true }
                                                        launchSingleTop = true
                                                    }
                                                }
                                            )
                                        }
                                        composable(CHAT_SCREEN) {
                                            ChatRoute()
                                        }
                                    }
                                }
                            }
                            else {
                                Button(onClick = {startLLM.value = true}, modifier = Modifier.align(Alignment.Center)) {
                                    Text(text = "Start LLM")
                                }
                            }
                        } //Chat UI
                        HorizontalDivider(color = MaterialTheme.colorScheme.outline, thickness = 3.dp)
                        Box (modifier = Modifier.fillMaxSize()){
                            // Conditional UI rendering based on camera permission state
                            if (startPreview.value) {
                                // Check Permission
                                handleCameraPermission()
                                CameraScreenWithLatencyLogging()
                            } else {
                                Button(
                                    onClick = {startPreview.value = true},modifier = Modifier.align(Alignment.Center)) {
                                    Text(text = "Start Preview")
                                }
                            }
                        } // Camera UI
                    }
                }
            }
        }
    }

    // TopAppBar is marked as experimental in Material 3
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun AppBar() {
        Column(
            modifier = Modifier.fillMaxWidth()
        ) {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name)) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.primary,
                ),
            )
            Box(
                modifier = Modifier.background(MaterialTheme.colorScheme.surfaceContainer)
            ) {
                Text(
                    text = stringResource(R.string.disclaimer),
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.fillMaxWidth()
                        .padding(8.dp)
                )
            }
        }
    }


    // Declare a launcher for the camera permission request, handling the permission result
    private val cameraPermissionRequestLauncher: ActivityResultLauncher<String> =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (!isGranted) {
                // Permission denied: inform the user to enable it through settings
                Toast.makeText(
                    this,
                    "Go to settings and enable camera permission to use this feature",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

    // Checks camera permission and either starts the camera directly or requests permission
    private fun handleCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                CAMERA
            ) != PackageManager.PERMISSION_GRANTED -> {
                // Permission is not granted: request it
                cameraPermissionRequestLauncher.launch(CAMERA)
            }
        }
    }


}
